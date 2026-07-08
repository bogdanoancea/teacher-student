"""
Supplemental experiments for the CLASSIFICATION knowledge-distillation study.
================================================================================
Companion to this script: `classification.py`. Run this script as:

  python classification_revision.py --quick   # 3 seeds (smoke test)
  python classification_revision.py           # 10 seeds
  python classification_revision.py --full    # 20 seeds (paper run)
  python classification_revision.py --xl      # 30 seeds (seed-sufficiency check)
  python classification_revision.py --full --only C4,C5

Experiments (map to reviewer comments):
  C1  Teacher-quality controls: oracle / single / homo / hetero / shuffled  (R1.1, R1.8)
  C2  Epistemic vs aleatoric uncertainty (entropy decomposition)            (R1.2)
  C3  Harder data: correlated / imbalance / class-dependent noise / skew    (R1.3, R2.2)
  C4  Calibration suite: ECE + Brier + ACE + MCE + NLL                      (R1.13)
  C5  Dark knowledge & soft->hard information loss + embeddings             (R1.15, R2.8)
  C6  alpha x T factorial DoE (main effects + interaction)                  (R1.4)
  C7  Robustness: covariate shift + FGSM + label-shift                      (R1.11)
  C8  Modern tabular models: TabPFN / FT-Transformer / TabNet (guarded)     (R1.5)
  C9  Extra baselines: mixup / born-again / temperature-only / KD           (R2.4)
  C10 Additional real datasets incl. imbalanced (OpenML, guarded)           (R1.14)
  C11 Seed sufficiency, effect sizes (Cohen d), McNemar                     (R1.7, R1.6)

Numeric output -> results_classification_revision_full.json
Requires Python 3.9+ (uses dict-union in a couple of summaries).
"""

import os, sys, time, json, warnings, platform
# Limit competing native thread pools on macOS/Apple Silicon. These must be set
# before NumPy, TensorFlow, PyTorch, XGBoost, or SciPy are imported.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from classification import (
    N_SAMPLES, N_CLASSES, N_FEATURES, BASE_ETA, ALPHAS, TEMPERATURES,
    MLP_EPOCHS, MLP_BATCH, TEMP_GRID, TEACHER_CONFIGS, MLP_CONFIGS,
    create_teacher_nn, create_small_mlp, get_sklearn_students,
    train_teacher_ensemble, train_mlp_distilled, compute_ece,
    aggregate_seeds, print_table,
)

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

# ============================================================
# CONFIG
# ============================================================
if '--xl' in sys.argv:
    MODE, N_SEEDS = 'XL', 30
elif '--full' in sys.argv:
    MODE, N_SEEDS = 'FULL', 20
elif '--quick' in sys.argv:
    MODE, N_SEEDS = 'QUICK', 3
else:
    MODE, N_SEEDS = 'DEFAULT', 10
SEEDS = [42 + i for i in range(N_SEEDS)]

def _only():
    if '--only' in sys.argv:
        return set(s.strip().upper() for s in sys.argv[sys.argv.index('--only') + 1].split(','))
    return None
ONLY = _only()
DUMP = {'config': {'mode': MODE, 'n_seeds': N_SEEDS, 'seeds': SEEDS}}


# ============================================================
# DATA GENERATOR WITH TRUE POSTERIORS (needed for the oracle teacher)
# ============================================================
def _logits(X):
    L = np.zeros((X.shape[0], N_CLASSES))
    L[:, 0] = 2.0*X[:, 0]**2 - 1.5*X[:, 1] + 0.5*np.sin(np.pi*X[:, 2])
    L[:, 1] = -X[:, 0] + 2.0*np.sin(np.pi*X[:, 1]) + X[:, 2]*X[:, 3]
    L[:, 2] = 1.5*X[:, 1]*X[:, 2] - X[:, 0]**2 + 0.8*np.exp(-X[:, 3]**2)
    L[:, 3] = -0.5*X[:, 2]**2 + 1.5*X[:, 3] + np.sin(np.pi*X[:, 4])*X[:, 0]
    L[:, 4] = X[:, 4]**2 - X[:, 5]*X[:, 0] + 0.5*np.cos(np.pi*X[:, 1])
    e = np.exp(L - L.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def gen(seed, eta=BASE_ETA, condition='base'):
    """Mirror classification.generate_synthetic but also return true posteriors
    p_true, and support harder 'condition' variants."""
    rng = np.random.RandomState(seed)
    n, d = N_SAMPLES, N_FEATURES
    if condition == 'correlated':
        cov = np.full((d, d), 0.6) + 0.4 * np.eye(d)
        X = rng.multivariate_normal(np.zeros(d), cov, size=n)
    elif condition == 'non_gaussian':
        X = rng.exponential(1.0, size=(n, d)) - 1.0
    else:
        X = rng.randn(n, d)

    p_true = _logits(X)
    y_true = p_true.argmax(axis=1)
    y = y_true.copy()

    if condition == 'class_noise':       # class-dependent (non-stationary) noise
        rates = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
        for i in range(n):
            if rng.random() < rates[y_true[i]]:
                y[i] = rng.choice([c for c in range(N_CLASSES) if c != y_true[i]])
    elif eta > 0:
        nf = int(eta * n)
        idx = rng.choice(n, size=nf, replace=False)
        for i in idx:
            y[i] = rng.choice([c for c in range(N_CLASSES) if c != y_true[i]])

    if condition == 'imbalance':         # drop samples to create skewed priors
        keep = np.array([1.0, 0.6, 0.4, 0.25, 0.15])
        mask = rng.random(n) < keep[y_true]
        X, y, y_true, p_true = X[mask], y[mask], y_true[mask], p_true[mask]

    X = StandardScaler().fit_transform(X)
    strat = y if condition != 'imbalance' else None
    Xtr, Xte, ytr, yte, pt_tr, pt_te, yt_tr, yt_te = train_test_split(
        X, y, p_true, y_true, test_size=0.2, random_state=seed, stratify=strat)
    strat2 = ytr if condition != 'imbalance' else None
    Xtr, Xva, ytr, yva, pt_tr, pt_va, yt_tr, yt_va = train_test_split(
        Xtr, ytr, pt_tr, yt_tr, test_size=0.15, random_state=seed, stratify=strat2)
    return dict(X_tr=Xtr, X_va=Xva, X_te=Xte, y_tr=ytr, y_va=yva, y_te=yte,
                yt_tr=yt_tr, yt_te=yt_te, p_true_tr=pt_tr, p_true_te=pt_te,
                n_features=X.shape[1], n_classes=N_CLASSES)


# ============================================================
# CALIBRATION METRICS
# ============================================================
def calib_metrics(y_true, probs, n_bins=15):
    conf = probs.max(axis=1); pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); ece = 0.0; mce = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() > 0:
            gap = abs(acc[m].mean() - conf[m].mean())
            ece += m.sum() / len(y_true) * gap
            mce = max(mce, gap)
    order = np.argsort(conf); ace = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            ace += len(chunk) / len(y_true) * abs(acc[chunk].mean() - conf[chunk].mean())
    oh = np.eye(probs.shape[1])[y_true]
    brier = float(np.mean(np.sum((probs - oh) ** 2, axis=1)))
    nll = float(log_loss(y_true, np.clip(probs, 1e-12, 1), labels=list(range(probs.shape[1]))))
    return {'ECE': float(ece), 'MCE': float(mce), 'ACE': float(ace),
            'Brier': brier, 'NLL': nll}


# ============================================================
# STUDENT DISTILLATION (hard pseudo-labels for sklearn, soft for MLP)
# ============================================================
def distill_students_clf(data, seed, soft_tr, alphas=ALPHAS, temps=TEMPERATURES,
                         include_sklearn=True, include_mlp=True):
    Xtr, Xva, Xte = data['X_tr'], data['X_va'], data['X_te']
    ytr, yva, yte = data['y_tr'], data['y_va'], data['y_te']
    nf, nc = data['n_features'], data['n_classes']
    hard_tr = soft_tr.argmax(axis=1)
    results = {}

    if include_sklearn:
        for name, model in get_sklearn_students(seed).items():
            results[name] = {}
            for a in alphas:
                m = clone(model)
                if a == 1.0:
                    m.fit(Xtr, ytr)
                elif a == 0.0:
                    m.fit(Xtr, hard_tr)
                else:
                    rng = np.random.RandomState(seed)
                    use_t = rng.random(len(ytr)) > a
                    m.fit(Xtr, np.where(use_t, hard_tr, ytr))
                results[name][a] = {'val': accuracy_score(yva, m.predict(Xva)),
                                    'test': accuracy_score(yte, m.predict(Xte))}
    if include_mlp:
        for name, (u, l) in MLP_CONFIGS.items():
            results[name] = {}
            tf.random.set_seed(seed)
            m = create_small_mlp(u, l, nf, nc)
            m.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            m.fit(Xtr.astype(np.float32), ytr.astype(np.int64),
                  epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            results[name][1.0] = {'val': accuracy_score(yva, m.predict(Xva, verbose=0).argmax(1)),
                                  'test': accuracy_score(yte, m.predict(Xte, verbose=0).argmax(1))}
            for a in [x for x in alphas if x != 1.0]:
                bv, bt = 0, 0
                for T in temps:
                    v, t, _ = train_mlp_distilled(Xtr, ytr, soft_tr, Xva, yva, Xte, yte,
                                                  u, l, nf, nc, a, T, seed)
                    if v > bv:
                        bv, bt = v, t
                results[name][a] = {'val': bv, 'test': bt}
    return results


def make_meta_clf(data, soft_tr, soft_te, members_tr=None, teacher_time=0.0):
    yte = data['y_te']
    acc = accuracy_score(yte, soft_te.argmax(1))
    ll = log_loss(yte, np.clip(soft_te, 1e-12, 1), labels=list(range(data['n_classes'])))
    ece = compute_ece(yte, soft_te)
    ent = float(-np.sum(soft_tr * np.log(soft_tr + 1e-10), axis=1).mean())
    corr = None
    if data.get('yt_tr') is not None:
        corr = accuracy_score(data['yt_tr'], soft_tr.argmax(1)) - accuracy_score(data['yt_tr'], data['y_tr'])
    return {'teacher_acc': acc, 'teacher_ll': ll, 'teacher_ece': ece,
            'teacher_entropy': ent, 'correction_rate': corr,
            'teacher_time': teacher_time, 'student_timing': {}}


def summarise(agg):
    models = {k: v for k, v in agg.items() if not k.startswith('_')}
    helped = sum(1 for v in models.values() if v['delta_pp'] > 0)
    avg = float(np.mean([v['delta_pp'] for v in models.values()]))
    mlp = float(np.mean([v['delta_pp'] for k, v in models.items() if 'MLP' in k]))
    return {'helped': helped, 'n_models': len(models),
            'avg_delta_pp': avg, 'mlp_delta_pp': mlp}


def cohens_dz(d):
    d = np.asarray(d, float); sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


# ============================================================
# TEACHER BUILDERS
# ============================================================
def build_teacher_clf(data, seed, kind, cfg_name='base'):
    Xtr, Xte = data['X_tr'], data['X_te']
    ytr = data['y_tr']; cfg = TEACHER_CONFIGS[cfg_name]
    nf, nc = data['n_features'], data['n_classes']

    if kind == 'oracle':
        return data['p_true_tr'], data['p_true_te'], [data['p_true_tr']], {}
    if kind in ('hetero', 'nn', 'xgb', 'gb'):
        pe, pn, px, pg = train_teacher_ensemble(data, cfg, seed)
        fn = {'hetero': pe, 'nn': pn, 'xgb': px, 'gb': pg}[kind]
        members = [pn(Xtr), px(Xtr), pg(Xtr)] if kind == 'hetero' else [fn(Xtr)]
        return fn(Xtr), fn(Xte), members, {}
    if kind == 'shuffled':
        pe, *_ = train_teacher_ensemble(data, cfg, seed)
        s_tr = pe(Xtr); rng = np.random.RandomState(seed + 777)
        return s_tr[rng.permutation(len(s_tr))], pe(Xte), [s_tr], {}
    if kind.startswith('homo_'):
        base = kind.split('_', 1)[1]; mt, me = [], []
        for j in range(3):
            sj = seed + 1000 * (j + 1)
            if base == 'nn':
                tf.random.set_seed(sj)
                nn = create_teacher_nn(cfg['nn_layers'], nf, nc)
                nn.compile(optimizer=keras.optimizers.Adam(0.001),
                           loss='sparse_categorical_crossentropy')
                nn.fit(Xtr, ytr, epochs=cfg['nn_epochs'], batch_size=32,
                       validation_data=(data['X_va'], data['y_va']),
                       callbacks=[keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True),
                                  keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8)], verbose=0)
                mt.append(nn.predict(Xtr, verbose=0)); me.append(nn.predict(Xte, verbose=0))
            elif base == 'xgb':
                m = XGBClassifier(n_estimators=cfg['xgb_n'], max_depth=cfg['xgb_d'],
                                  learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                  num_class=nc, objective='multi:softprob',
                                  random_state=sj, verbosity=0, use_label_encoder=False)
                m.fit(Xtr, ytr); mt.append(m.predict_proba(Xtr)); me.append(m.predict_proba(Xte))
            elif base == 'gb':
                m = GradientBoostingClassifier(n_estimators=cfg['gb_n'], max_depth=cfg['gb_d'],
                                               learning_rate=0.05, subsample=0.8, random_state=sj)
                m.fit(Xtr, ytr); mt.append(m.predict_proba(Xtr)); me.append(m.predict_proba(Xte))
        return np.mean(mt, axis=0), np.mean(me, axis=0), mt, {}
    raise ValueError(kind)


# ============================================================
# C1  TEACHER-QUALITY CONTROLS   (R1.1, R1.8)
# ============================================================
def exp_C1():
    print("\n" + "=" * 80); print(f"C1  TEACHER-QUALITY CONTROLS (seeds={N_SEEDS})"); print("=" * 80)
    kinds = ['oracle', 'hetero', 'nn', 'xgb', 'gb', 'homo_nn', 'homo_xgb', 'homo_gb', 'shuffled']
    if MODE == 'QUICK':
        kinds = ['oracle', 'hetero', 'gb', 'homo_gb', 'shuffled']
    out = {}
    for kind in kinds:
        print(f"\n--- teacher = {kind} ---")
        all_data = []
        for s, seed in enumerate(SEEDS):
            print(f"  seed {s+1}/{N_SEEDS}", end=" ", flush=True); t0 = time.time()
            data = gen(seed, BASE_ETA)
            s_tr, s_te, members, _ = build_teacher_clf(data, seed, kind)
            res = distill_students_clf(data, seed, s_tr)
            all_data.append((res, make_meta_clf(data, s_tr, s_te, members)))
            print(f"({time.time()-t0:.0f}s)", end="")
        print()
        agg = aggregate_seeds(all_data); sm = summarise(agg)
        sm['teacher_acc'] = float(agg['_teacher']['acc_mean'])
        sm['teacher_ece'] = float(agg['_teacher']['ece_mean'])
        sm['teacher_entropy'] = float(agg['_teacher']['entropy_mean'])
        sm['correction'] = agg['_teacher']['correction']
        out[kind] = {'summary': sm, 'aggregate': agg}
        print(f"  acc={sm['teacher_acc']:.3f} ECE={sm['teacher_ece']:.3f} "
              f"ent={sm['teacher_entropy']:.2f} corr={sm['correction']} "
              f"helped={sm['helped']}/{sm['n_models']} MLP={sm['mlp_delta_pp']:+.2f}pp")
    print("\n" + "-" * 80)
    print(f"{'Teacher':<11} {'Acc':<7} {'ECE':<7} {'Entropy':<8} {'Corr%':<7} {'Helped':<8} {'Avg pp':<8} {'MLP pp':<8}")
    print("-" * 80)
    for k in kinds:
        s = out[k]['summary']
        cr = f"{s['correction']:.1f}" if s['correction'] is not None else " - "
        print(f"{k:<11} {s['teacher_acc']:<7.3f} {s['teacher_ece']:<7.3f} {s['teacher_entropy']:<8.2f} "
              f"{cr:<7} {s['helped']}/{s['n_models']:<6} {s['avg_delta_pp']:<+8.2f} {s['mlp_delta_pp']:<+8.2f}")
    print("\nReading: oracle (=Bayes posterior) is the ability ceiling; homo_* isolate "
          "variance reduction within one hypothesis class; shuffled keeps the teacher's "
          "marginal but destroys per-sample ability. If gains track accuracy and collapse "
          "for 'shuffled', the benefit is teacher ABILITY, not ensemble variance alone.")
    DUMP['C1'] = {k: out[k]['summary'] for k in out}
    DUMP['C1_full'] = {k: out[k]['aggregate'] for k in out}
    return out


# ============================================================
# C2  UNCERTAINTY DECOMPOSITION (entropy)   (R1.2)
# ============================================================
def exp_C2():
    print("\n" + "=" * 80); print(f"C2  UNCERTAINTY DECOMPOSITION (seeds={N_SEEDS})"); print("=" * 80)
    rows = []
    for s, seed in enumerate(SEEDS):
        print(f"  seed {s+1}/{N_SEEDS}", end=" ", flush=True); t0 = time.time()
        data = gen(seed, BASE_ETA)
        pe, pn, px, pg = train_teacher_ensemble(data, TEACHER_CONFIGS['base'], seed)
        Xtr = data['X_tr']
        members = np.stack([pn(Xtr), px(Xtr), pg(Xtr)], axis=0)   # (3, n, K)
        mean_p = members.mean(axis=0)
        H = lambda p: -np.sum(p * np.log(p + 1e-10), axis=-1)
        total = H(mean_p)                                         # predictive entropy
        aleatoric = H(members).mean(axis=0)                       # expected member entropy
        epistemic = total - aleatoric                             # mutual information
        rows.append({'seed': seed, 'total': float(total.mean()),
                     'aleatoric': float(aleatoric.mean()),
                     'epistemic_MI': float(epistemic.mean()),
                     'epi_fraction': float(epistemic.mean() / (total.mean() + 1e-12))})
        print(f"({time.time()-t0:.0f}s)", end="")
    print()
    tot = np.mean([r['total'] for r in rows]); ale = np.mean([r['aleatoric'] for r in rows])
    epi = np.mean([r['epistemic_MI'] for r in rows])
    print(f"\n  total predictive entropy : {tot:.3f} nats")
    print(f"  aleatoric (E[H(member)]) : {ale:.3f} nats")
    print(f"  epistemic (MI)           : {epi:.3f} nats  ({epi/tot*100:.1f}% of total)")
    DUMP['C2'] = {'per_seed': rows, 'total': float(tot), 'aleatoric': float(ale), 'epistemic_MI': float(epi)}
    return rows


# ============================================================
# C3  HARDER DATA CONDITIONS   (R1.3)
# ============================================================
def exp_C3():
    print("\n" + "=" * 80); print(f"C3  HARDER DATA CONDITIONS (seeds={N_SEEDS})"); print("=" * 80)
    conds = ['correlated', 'imbalance', 'class_noise', 'non_gaussian']
    if MODE == 'QUICK':
        conds = ['correlated', 'imbalance']
    out = {}
    for cond in conds:
        print(f"\n--- condition = {cond} ---")
        all_data = []
        for s, seed in enumerate(SEEDS):
            print(f"  seed {s+1}/{N_SEEDS}", end=" ", flush=True); t0 = time.time()
            data = gen(seed, BASE_ETA, condition=cond)
            pe, pn, px, pg = train_teacher_ensemble(data, TEACHER_CONFIGS['base'], seed)
            s_tr, s_te = pe(data['X_tr']), pe(data['X_te'])
            res = distill_students_clf(data, seed, s_tr)
            all_data.append((res, make_meta_clf(data, s_tr, s_te,
                                                [pn(data['X_tr']), px(data['X_tr']), pg(data['X_tr'])])))
            print(f"({time.time()-t0:.0f}s)", end="")
        print()
        agg = aggregate_seeds(all_data); sm = summarise(agg)
        sm['teacher_acc'] = float(agg['_teacher']['acc_mean'])
        out[cond] = {'summary': sm, 'aggregate': agg}
        print(f"  teacher acc={sm['teacher_acc']:.3f} helped={sm['helped']}/{sm['n_models']} "
              f"avg={sm['avg_delta_pp']:+.2f}pp MLP={sm['mlp_delta_pp']:+.2f}pp")
    DUMP['C3'] = {c: out[c]['summary'] for c in out}
    DUMP['C3_full'] = {c: out[c]['aggregate'] for c in out}
    return out


# ============================================================
# C4  CALIBRATION SUITE   (R1.13)
# ============================================================
def exp_C4():
    print("\n" + "=" * 80); print(f"C4  CALIBRATION SUITE: ECE/MCE/ACE/Brier/NLL (seeds={N_SEEDS})"); print("=" * 80)
    keys = ['teacher', 'baseline_MLP(32x2)', 'distil_MLP(32x2)', 'baseline_MLP(8)', 'distil_MLP(8)']
    store = {k: [] for k in keys}
    for s, seed in enumerate(SEEDS):
        print(f"  seed {s+1}/{N_SEEDS}", end=" ", flush=True); t0 = time.time()
        data = gen(seed, BASE_ETA)
        Xtr, ytr, Xva, yva, Xte, yte = (data['X_tr'], data['y_tr'], data['X_va'],
                                        data['y_va'], data['X_te'], data['y_te'])
        nf, nc = data['n_features'], data['n_classes']
        pe, *_ = train_teacher_ensemble(data, TEACHER_CONFIGS['base'], seed)
        store['teacher'].append(calib_metrics(yte, pe(Xte)))
        s_tr = pe(Xtr)
        for mname, (u, l) in [('MLP(32x2)', (32, 2)), ('MLP(8)', (8, 1))]:
            tf.random.set_seed(seed)
            m = create_small_mlp(u, l, nf, nc)
            m.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            m.fit(Xtr.astype(np.float32), ytr.astype(np.int64), epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            store[f'baseline_{mname}'].append(calib_metrics(yte, tf.nn.softmax(m.predict(Xte, verbose=0)).numpy()))
            _, _, dprobs = train_mlp_distilled(Xtr, ytr, s_tr, Xva, yva, Xte, yte, u, l, nf, nc, 0.3, 5.0, seed)
            store[f'distil_{mname}'].append(calib_metrics(yte, dprobs))
        print(f"({time.time()-t0:.0f}s)", end="")
    print()
    metrics = ['ECE', 'MCE', 'ACE', 'Brier', 'NLL']
    print(f"\n{'Model':<22}" + "".join(f"{m:<9}" for m in metrics))
    print("-" * (22 + 9 * len(metrics)))
    summ = {}
    for k in keys:
        summ[k] = {m: float(np.mean([d[m] for d in store[k]])) for m in metrics}
        print(f"{k:<22}" + "".join(f"{summ[k][m]:<9.4f}" for m in metrics))
    DUMP['C4'] = summ
    return summ


# ============================================================
# C5  DARK KNOWLEDGE & SOFT->HARD INFORMATION LOSS + EMBEDDINGS   (R1.15, R2.8)
# ============================================================
def _embedding_model(u, l, nf, nc):
    inp = keras.Input(shape=(nf,)); x = inp
    for _ in range(l):
        x = layers.Dense(u, activation='relu')(x)
    out = layers.Dense(nc)(x)
    return keras.Model(inp, out), keras.Model(inp, x)   # (full, penultimate)


def _train_soft_mlp(full, Xtr, ytr, soft, seed, alpha=0.3, T=5.0):
    opt = keras.optimizers.Adam(0.001)
    ds = tf.data.Dataset.from_tensor_slices(
        (Xtr.astype(np.float32), ytr.astype(np.int64), soft.astype(np.float32))
    ).shuffle(2048, seed=seed).batch(MLP_BATCH)
    for _ in range(MLP_EPOCHS):
        for xb, yb, sb in ds:
            with tf.GradientTape() as tape:
                lo = full(xb, training=True)
                hl = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(yb, tf.nn.softmax(lo)))
                sp = tf.nn.softmax(lo / T); st = tf.nn.softmax(tf.math.log(sb + 1e-10) / T)
                loss = alpha * hl + (1 - alpha) * (T ** 2) * keras.losses.KLDivergence()(st, sp)
            g = tape.gradient(loss, full.trainable_variables)
            opt.apply_gradients(zip(g, full.trainable_variables))


def exp_C5():
    print("\n" + "=" * 80); print(f"C5  DARK KNOWLEDGE & INFO LOSS (seeds={N_SEEDS})"); print("=" * 80)
    info_rows = []; sil = {'baseline': [], 'distilled': []}
    for s, seed in enumerate(SEEDS):
        print(f"  seed {s+1}/{N_SEEDS}", end=" ", flush=True); t0 = time.time()
        data = gen(seed, BASE_ETA)
        Xtr, ytr, Xte, yte = data['X_tr'], data['y_tr'], data['X_te'], data['y_te']
        nf, nc = data['n_features'], data['n_classes']
        pe, *_ = train_teacher_ensemble(data, TEACHER_CONFIGS['base'], seed)
        soft = pe(Xtr)
        ent = -np.sum(soft * np.log(soft + 1e-10), axis=1)
        srt = np.sort(soft, axis=1)
        dark_mass = 1.0 - srt[:, -1]
        second_mass = srt[:, -2]
        onehot = np.eye(nc)[soft.argmax(1)]
        kl_soft_hard = np.sum(soft * (np.log(soft + 1e-10) - np.log(onehot + 1e-10)), axis=1)
        info_rows.append({'seed': seed,
                          'mean_entropy_nats': float(ent.mean()),
                          'mean_dark_mass': float(dark_mass.mean()),
                          'mean_second_class_mass': float(second_mass.mean()),
                          'mean_KL_soft_to_hard': float(kl_soft_hard.mean()),
                          'frac_secondmass_gt_0.1': float((second_mass > 0.1).mean())})
        for tag, alpha in [('baseline', 1.0), ('distilled', 0.3)]:
            tf.random.set_seed(seed)
            full, emb = _embedding_model(32, 2, nf, nc)
            if alpha == 1.0:
                full.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
                full.fit(Xtr.astype(np.float32), ytr.astype(np.int64), epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            else:
                _train_soft_mlp(full, Xtr, ytr, soft, seed)
            feats = emb.predict(Xte, verbose=0)
            idx = np.random.RandomState(0).choice(len(feats), min(800, len(feats)), replace=False)
            try:
                sil[tag].append(float(silhouette_score(feats[idx], yte[idx])))
            except Exception:
                sil[tag].append(float('nan'))
        print(f"({time.time()-t0:.0f}s)", end="")
    print()
    agg_info = {k: float(np.mean([r[k] for r in info_rows])) for k in info_rows[0] if k != 'seed'}
    print("\n  Soft-target information (mean over seeds):")
    for k, v in agg_info.items():
        print(f"    {k:<28} {v:.4f}")
    print(f"\n  Penultimate-layer silhouette (class separation on test):")
    print(f"    baseline  : {np.nanmean(sil['baseline']):.4f}")
    print(f"    distilled : {np.nanmean(sil['distilled']):.4f}")
    print("  Interpretation: KL(soft||argmax-onehot) is exactly the information a hard "
          "pseudo-label discards. Non-neural students only ever see the argmax, so they "
          "cannot receive this dark-knowledge mass.")
    DUMP['C5'] = {'soft_info': agg_info, 'per_seed_info': info_rows,
                  'silhouette_baseline': float(np.nanmean(sil['baseline'])),
                  'silhouette_distilled': float(np.nanmean(sil['distilled']))}
    return DUMP['C5']


# ============================================================
# C6  alpha x T FACTORIAL DoE   (R1.4)
# ============================================================
def exp_C6():
    print("\n" + "=" * 80); print(f"C6  alpha x T FACTORIAL DoE for MLP(32x2) (seeds={N_SEEDS})"); print("=" * 80)
    A = [0.0, 0.3, 0.5, 0.7]; Tg = TEMP_GRID
    grid = {(a, T): [] for a in A for T in Tg}
    for s, seed in enumerate(SEEDS):
        print(f"  seed {s+1}/{N_SEEDS}", end=" ", flush=True); t0 = time.time()
        data = gen(seed, BASE_ETA)
        pe, *_ = train_teacher_ensemble(data, TEACHER_CONFIGS['base'], seed)
        s_tr = pe(data['X_tr'])
        for a in A:
            for T in Tg:
                _, te, _ = train_mlp_distilled(data['X_tr'], data['y_tr'], s_tr,
                                               data['X_va'], data['y_va'], data['X_te'], data['y_te'],
                                               32, 2, data['n_features'], data['n_classes'], a, T, seed)
                grid[(a, T)].append(te)
        print(f"({time.time()-t0:.0f}s)", end="")
    print()
    mean = {k: float(np.mean(v)) for k, v in grid.items()}
    print(f"\n  Mean test accuracy grid (rows=alpha, cols=T):")
    print("  alpha\\T   " + "  ".join(f"{T:>6.0f}" for T in Tg))
    for a in A:
        print(f"   {a:<6} " + "  ".join(f"{mean[(a,T)]:.4f}" for T in Tg))
    Y = np.array([[np.array(grid[(a, T)]) for T in Tg] for a in A])  # (|A|,|T|,S)
    grand = Y.mean()
    a_eff = Y.mean(axis=(1, 2)) - grand
    t_eff = Y.mean(axis=(0, 2)) - grand
    inter = Y.mean(axis=2) - (grand + a_eff[:, None] + t_eff[None, :])
    S = Y.shape[2]
    SS_a = S * len(Tg) * np.sum(a_eff ** 2)
    SS_t = S * len(A) * np.sum(t_eff ** 2)
    SS_i = S * np.sum(inter ** 2)
    SS_e = np.sum((Y - Y.mean(axis=2, keepdims=True)) ** 2)
    df_a, df_t = len(A) - 1, len(Tg) - 1
    df_i = df_a * df_t; df_e = len(A) * len(Tg) * (S - 1)
    def F(ss, df): return (ss / df) / (SS_e / df_e) if df_e > 0 else float('nan')
    anova = {'F_alpha': float(F(SS_a, df_a)), 'F_T': float(F(SS_t, df_t)),
             'F_interaction': float(F(SS_i, df_i)),
             'SS_alpha': float(SS_a), 'SS_T': float(SS_t),
             'SS_interaction': float(SS_i), 'SS_resid': float(SS_e)}
    print(f"\n  Two-way ANOVA: F(alpha)={anova['F_alpha']:.1f}  F(T)={anova['F_T']:.1f}  "
          f"F(alpha x T)={anova['F_interaction']:.2f}")
    DUMP['C6'] = {'grid_mean': {f"a{a}_T{T}": mean[(a, T)] for a in A for T in Tg}, 'anova': anova}
    return DUMP['C6']


# ============================================================
# C7  ROBUSTNESS: covariate shift + FGSM   (R1.11)
# ============================================================
def exp_C7():
    print("\n" + "=" * 80); print(f"C7  ROBUSTNESS (seeds={N_SEEDS})"); print("=" * 80)
    targets = ['MLP(8)', 'MLP(32x2)']
    conds = ['clean', 'mean_shift', 'scale_shift', 'fgsm']
    store = {t: {c: {'base': [], 'distil': []} for c in conds} for t in targets}
    for s, seed in enumerate(SEEDS):
        print(f"  seed {s+1}/{N_SEEDS}", end=" ", flush=True); t0 = time.time()
        data = gen(seed, BASE_ETA)
        Xtr, ytr, Xte, yte = data['X_tr'], data['y_tr'], data['X_te'], data['y_te']
        nf, nc = data['n_features'], data['n_classes']
        pe, *_ = train_teacher_ensemble(data, TEACHER_CONFIGS['base'], seed)
        s_tr = pe(Xtr)
        for t in targets:
            u, l = MLP_CONFIGS[t]; models = {}
            tf.random.set_seed(seed)
            mb = create_small_mlp(u, l, nf, nc)
            mb.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            mb.fit(Xtr.astype(np.float32), ytr.astype(np.int64), epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            models['base'] = mb
            tf.random.set_seed(seed)
            md, _ = _embedding_model(u, l, nf, nc)
            _train_soft_mlp(md, Xtr, ytr, s_tr, seed)
            models['distil'] = md
            for tag, m in models.items():
                for c in conds:
                    if c == 'clean':
                        Xev = Xte
                    elif c == 'mean_shift':
                        Xev = Xte + 0.5
                    elif c == 'scale_shift':
                        Xev = Xte * 1.5
                    else:  # fgsm
                        xv = tf.convert_to_tensor(Xte.astype(np.float32))
                        with tf.GradientTape() as tp:
                            tp.watch(xv)
                            ce = keras.losses.sparse_categorical_crossentropy(
                                yte.astype(np.int64), tf.nn.softmax(m(xv)))
                        grad = tp.gradient(ce, xv)
                        Xev = (xv + 0.1 * tf.sign(grad)).numpy()
                    store[t][c][tag].append(accuracy_score(yte, m.predict(Xev, verbose=0).argmax(1)))
        print(f"({time.time()-t0:.0f}s)", end="")
    print()
    print(f"\n{'Model':<10} {'Condition':<12} {'Base':<9} {'Distil':<9} {'D(pp)':<8}")
    print("-" * 50)
    summ = {}
    for t in targets:
        summ[t] = {}
        for c in conds:
            b = float(np.mean(store[t][c]['base'])); d = float(np.mean(store[t][c]['distil']))
            summ[t][c] = {'base': b, 'distil': d, 'delta_pp': (d - b) * 100}
            print(f"{t:<10} {c:<12} {b:<9.4f} {d:<9.4f} {(d-b)*100:<+8.2f}")
    DUMP['C7'] = summ
    return summ


# ============================================================
# C8  MODERN TABULAR MODELS (guarded)   (R1.5)
# ============================================================
def exp_C8():
    print("\n" + "=" * 80); print(f"C8  MODERN TABULAR MODELS (guarded) (seeds={min(N_SEEDS,5)})"); print("=" * 80, flush=True)
    n = min(N_SEEDS, 5)
    out = {'available': {}, 'results': {}}

    skip_tabpfn = '--skip-tabpfn' in sys.argv
    skip_pt = '--skip-pytorch-tabular' in sys.argv

    # Native-extension crashes (SIGSEGV/SIGABRT) cannot be caught by try/except.
    # These progress messages identify the exact backend that fails.
    if skip_tabpfn:
        out['available']['tabpfn'] = False
        print("  [skip] TabPFN disabled by --skip-tabpfn", flush=True)
    else:
        print("  [probe] importing TabPFN ...", flush=True)
        try:
            from tabpfn import TabPFNClassifier
            print("  [probe] TabPFN import OK", flush=True)
            comp, as_teacher = [], []
            for s in range(n):
                seed = 42 + s
                print(f"  [TabPFN] seed {s+1}/{n}: preparing data", flush=True)
                data = gen(seed, BASE_ETA)
                Xtr, ytr, Xte, yte = data['X_tr'], data['y_tr'], data['X_te'], data['y_te']
                lim = min(1000, len(Xtr))
                idx = np.random.RandomState(seed).choice(len(Xtr), lim, replace=False)
                print(f"  [TabPFN] seed {s+1}/{n}: constructing model", flush=True)
                clf = TabPFNClassifier(device='cpu')
                print(f"  [TabPFN] seed {s+1}/{n}: fitting", flush=True)
                clf.fit(np.asarray(Xtr[idx], dtype=np.float32), np.asarray(ytr[idx], dtype=np.int64))
                comp.append(accuracy_score(yte, clf.predict(np.asarray(Xte, dtype=np.float32))))
                soft_full = clf.predict_proba(np.asarray(Xtr, dtype=np.float32))
                _, te, _ = train_mlp_distilled(Xtr, ytr, soft_full, data['X_va'], data['y_va'],
                                               Xte, yte, 32, 2, data['n_features'], data['n_classes'],
                                               0.3, 5.0, seed)
                as_teacher.append(te)
            out['available']['tabpfn'] = True
            out['results']['tabpfn'] = {'standalone_acc': float(np.mean(comp)),
                                        'as_teacher_MLP32x2_acc': float(np.mean(as_teacher))}
            print(f"  TabPFN: standalone acc={np.mean(comp):.4f}  as-teacher->MLP(32x2)={np.mean(as_teacher):.4f}", flush=True)
        except Exception as e:
            out['available']['tabpfn'] = False
            print(f"  [skip] TabPFN not available: {type(e).__name__}: {e}", flush=True)

    if skip_pt:
        out['available']['pytorch_tabular'] = False
        print("  [skip] pytorch-tabular disabled by --skip-pytorch-tabular", flush=True)
    else:
        print("  [probe] importing pytorch-tabular ...", flush=True)
        try:
            import pandas as pd
            from pytorch_tabular import TabularModel
            from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
            from pytorch_tabular.models import FTTransformerConfig, TabNetModelConfig
            print("  [probe] pytorch-tabular import OK", flush=True)
            for label, mkcfg in [('ft_transformer', lambda: FTTransformerConfig(task='classification')),
                                 ('tabnet', lambda: TabNetModelConfig(task='classification'))]:
                try:
                    accs = []
                    for s in range(n):
                        seed = 42 + s
                        print(f"  [{label}] seed {s+1}/{n}", flush=True)
                        data = gen(seed, BASE_ETA)
                        cols = [f'f{i}' for i in range(data['n_features'])]
                        tr = pd.DataFrame(data['X_tr'], columns=cols); tr['target'] = data['y_tr']
                        te = pd.DataFrame(data['X_te'], columns=cols); te['target'] = data['y_te']
                        try:
                            model_config = mkcfg()
                            tm = TabularModel(
                                data_config=DataConfig(target=['target'], continuous_cols=cols),
                                model_config=model_config, optimizer_config=OptimizerConfig(),
                                trainer_config=TrainerConfig(max_epochs=30, accelerator='cpu',
                                                             devices=1, progress_bar='none'))
                            tm.fit(train=tr)
                        except TypeError as config_error:
                            # Some pytorch-tabular / pytorch-tabnet combinations are
                            # API-incompatible (notably create_group_matrix).  For
                            # TabNet, use its maintained sklearn-style API directly.
                            if label != 'tabnet' or 'create_group_matrix' not in str(config_error):
                                raise
                            from pytorch_tabnet.tab_model import TabNetClassifier
                            direct = TabNetClassifier(seed=seed, verbose=0)
                            direct.fit(
                                data['X_tr'].astype(np.float32),
                                data['y_tr'].astype(np.int64),
                                eval_set=[(data['X_va'].astype(np.float32),
                                           data['y_va'].astype(np.int64))],
                                eval_name=['validation'],
                                eval_metric=['accuracy'],
                                max_epochs=30,
                                patience=8,
                                batch_size=min(1024, len(data['X_tr'])),
                                virtual_batch_size=min(128, len(data['X_tr'])),
                                num_workers=0,
                                drop_last=False,
                            )
                            y_hat = direct.predict(data['X_te'].astype(np.float32))
                            accs.append(accuracy_score(data['y_te'], y_hat))
                            continue
                        # Do not include the target in the prediction frame.  Recent
                        # pytorch-tabular versions name the class column
                        # ``target_prediction`` rather than simply ``prediction``.
                        pred = tm.predict(te.drop(columns=['target']))

                        prediction_cols = [c for c in pred.columns
                                           if c == 'prediction' or c.endswith('_prediction')]
                        if prediction_cols:
                            y_hat = pred[prediction_cols[0]].to_numpy()
                        else:
                            # Version-independent fallback: recover labels from the
                            # per-class probability columns returned by predict().
                            probability_cols = [c for c in pred.columns
                                                if 'probability' in str(c).lower()]
                            if len(probability_cols) >= data['n_classes']:
                                y_hat = pred[probability_cols].to_numpy().argmax(axis=1)
                            else:
                                raise KeyError(
                                    'No prediction column found. Returned columns: '
                                    + ', '.join(map(str, pred.columns))
                                )
                        accs.append(accuracy_score(data['y_te'], np.asarray(y_hat, dtype=int)))
                    out['available'][label] = True
                    out['results'][label] = {'standalone_acc': float(np.mean(accs))}
                    print(f"  {label}: standalone acc={np.mean(accs):.4f}", flush=True)
                except Exception as e:
                    out['available'][label] = False
                    print(f"  [skip] {label} run failed: {type(e).__name__}: {e}", flush=True)
        except Exception as e:
            out['available']['pytorch_tabular'] = False
            print(f"  [skip] pytorch-tabular (FT-Transformer/TabNet) not available: {type(e).__name__}: {e}", flush=True)

    print("  (SAINT is skipped by default to avoid a hard dependency; add a hook if installed.)", flush=True)
    DUMP['C8'] = out
    return out


# ============================================================
# C9  EXTRA BASELINES   (R2.4)
# ============================================================
def exp_C9():
    print("\n" + "=" * 80); print(f"C9  EXTRA BASELINES vs KD for MLPs (seeds={N_SEEDS})"); print("=" * 80)
    cols = ['standard', 'mixup', 'born_again', 'temp_only', 'KD']
    res = {name: {c: [] for c in cols} for name in MLP_CONFIGS}
    for s, seed in enumerate(SEEDS):
        print(f"  seed {s+1}/{N_SEEDS}", end=" ", flush=True); t0 = time.time()
        data = gen(seed, BASE_ETA)
        Xtr, ytr, Xva, yva, Xte, yte = (data['X_tr'], data['y_tr'], data['X_va'],
                                        data['y_va'], data['X_te'], data['y_te'])
        nf, nc = data['n_features'], data['n_classes']
        pe, *_ = train_teacher_ensemble(data, TEACHER_CONFIGS['base'], seed)
        s_tr = pe(Xtr); rng = np.random.RandomState(seed)
        for name, (u, l) in MLP_CONFIGS.items():
            # standard
            tf.random.set_seed(seed)
            m = create_small_mlp(u, l, nf, nc)
            m.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            m.fit(Xtr.astype(np.float32), ytr.astype(np.int64), epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            res[name]['standard'].append(accuracy_score(yte, m.predict(Xte, verbose=0).argmax(1)))
            # mixup (soft one-hot targets, trained with categorical CE)
            tf.random.set_seed(seed)
            lam = rng.beta(0.2, 0.2, size=len(Xtr))[:, None]; perm = rng.permutation(len(Xtr))
            Xm = lam * Xtr + (1 - lam) * Xtr[perm]
            oh = np.eye(nc)[ytr]; ym = lam * oh + (1 - lam) * oh[perm]
            m = create_small_mlp(u, l, nf, nc)
            m.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(from_logits=True))
            m.fit(Xm.astype(np.float32), ym.astype(np.float32), epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            res[name]['mixup'].append(accuracy_score(yte, m.predict(Xte, verbose=0).argmax(1)))
            # born-again self-distillation: train teacher MLP, distil into identical student
            tf.random.set_seed(seed)
            m1 = create_small_mlp(u, l, nf, nc)
            m1.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            m1.fit(Xtr.astype(np.float32), ytr.astype(np.int64), epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            self_soft = tf.nn.softmax(m1.predict(Xtr, verbose=0)).numpy()
            full, _ = _embedding_model(u, l, nf, nc)
            tf.random.set_seed(seed)
            _train_soft_mlp(full, Xtr, ytr, self_soft, seed, alpha=0.5, T=3.0)
            res[name]['born_again'].append(accuracy_score(yte, full.predict(Xte, verbose=0).argmax(1)))
            # temperature-only KD (alpha=0, T selected on val)
            bv, bt = 0, 0
            for T in [3.0, 5.0, 10.0]:
                v, t, _ = train_mlp_distilled(Xtr, ytr, s_tr, Xva, yva, Xte, yte, u, l, nf, nc, 0.0, T, seed)
                if v > bv:
                    bv, bt = v, t
            res[name]['temp_only'].append(bt)
            # full KD (alpha+T selected on val)
            bv, bt = 0, 0
            for a in [0.0, 0.3, 0.5, 0.7]:
                for T in TEMPERATURES:
                    v, t, _ = train_mlp_distilled(Xtr, ytr, s_tr, Xva, yva, Xte, yte, u, l, nf, nc, a, T, seed)
                    if v > bv:
                        bv, bt = v, t
            res[name]['KD'].append(bt)
        print(f"({time.time()-t0:.0f}s)", end="")
    print()
    print(f"\n{'Model':<10}" + "".join(f"{c:<13}" for c in cols))
    print("-" * (10 + 13 * len(cols)))
    summ = {}
    for name in MLP_CONFIGS:
        summ[name] = {c: float(np.mean(res[name][c])) for c in cols}
        print(f"{name:<10}" + "".join(f"{summ[name][c]:<13.4f}" for c in cols))
    DUMP['C9'] = summ
    return summ


# ============================================================
# C10  ADDITIONAL REAL DATASETS incl. imbalanced (OpenML, guarded)   (R1.14)
# ============================================================
def _split_real(X, y, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.15, random_state=seed, stratify=ytr)
    return dict(X_tr=Xtr, X_va=Xva, X_te=Xte, y_tr=ytr, y_va=yva, y_te=yte,
                yt_tr=None, yt_te=None, n_features=X.shape[1], n_classes=len(np.unique(y)))


def _load_openml_classification(did):
    """Fetch an OpenML classification dataset robustly -> clean 2-D X, 1-D int y."""
    import pandas as pd
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(data_id=did, as_frame=True)
    Xdf = ds.data.select_dtypes('number')                  # numeric features only
    X = np.asarray(Xdf, dtype=float)
    y = np.asarray(pd.Series(np.asarray(ds.target).ravel()).astype('category').cat.codes, dtype=int)
    keep = np.isfinite(X).all(axis=1) & (y >= 0)
    X, y = X[keep], y[keep]
    if X.ndim != 2 or X.shape[0] < 100 or X.shape[1] < 2:
        raise ValueError(f"unsuitable shape X={X.shape}")
    # need every class to have enough members for a stratified 3-way split
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 10 or len(counts) < 2:
        raise ValueError(f"too few members in smallest class (min={counts.min()})")
    X = StandardScaler().fit_transform(X)
    return X, y


def exp_C10():
    print("\n" + "=" * 80); print(f"C10  ADDITIONAL REAL DATASETS (OpenML; needs network) (seeds={min(N_SEEDS,10)})"); print("=" * 80)
    catalog = {'adult': 1590, 'credit-g': 31, 'electricity': 151}   # imbalanced / correlated
    n = min(N_SEEDS, 10); out = {}
    for dsname, did in catalog.items():
        print(f"\n--- {dsname} (openml {did}) ---")
        try:                                                # ENTIRE dataset guarded: fetch + all seeds
            X, y = _load_openml_classification(did)
            _, cnt = np.unique(y, return_counts=True)
            print(f"  loaded X={X.shape}, classdist={cnt.tolist()}")
            all_data = []
            for s in range(n):
                seed = 42 + s
                data = _split_real(X, y, seed)
                pe, *_ = train_teacher_ensemble(data, TEACHER_CONFIGS['base'], seed)
                s_tr, s_te = pe(data['X_tr']), pe(data['X_te'])
                res = distill_students_clf(data, seed, s_tr)
                all_data.append((res, make_meta_clf(data, s_tr, s_te)))
                print(f"  seed {s+1}/{n} done")
            agg = aggregate_seeds(all_data); sm = summarise(agg)
            sm['teacher_acc'] = float(agg['_teacher']['acc_mean'])
            out[dsname] = {'summary': sm, 'aggregate': agg}
            print(f"  teacher acc={sm['teacher_acc']:.3f} helped={sm['helped']}/{sm['n_models']} "
                  f"avg={sm['avg_delta_pp']:+.2f}pp MLP={sm['mlp_delta_pp']:+.2f}pp")
        except Exception as e:
            print(f"  [skip] {dsname}: {e}"); out[dsname] = {'skipped': str(e)}
    DUMP['C10'] = {k: (v.get('summary', v)) for k, v in out.items()}
    DUMP['C10_full'] = {k: v['aggregate'] for k, v in out.items() if 'aggregate' in v}
    return out


# ============================================================
# C11  SEED SUFFICIENCY, EFFECT SIZES, McNemar   (R1.7, R1.6)
# ============================================================
def exp_C11():
    print("\n" + "=" * 80); print(f"C11  SEED SUFFICIENCY / EFFECT SIZE / McNemar (seeds={N_SEEDS})"); print("=" * 80)
    targets = ['MLP(8)', 'MLP(16x2)', 'MLP(32x2)', 'MLP(32)', 'MLP(16)']
    per_seed = {t: [] for t in targets}
    # pooled prediction agreement for McNemar (baseline vs distilled), MLP(8)
    mcnemar = {'b_only': 0, 'd_only': 0}
    n_test_total = 0
    for s, seed in enumerate(SEEDS):
        print(f"  seed {s+1}/{N_SEEDS}", end=" ", flush=True); t0 = time.time()
        data = gen(seed, BASE_ETA)
        Xtr, ytr, Xva, yva, Xte, yte = (data['X_tr'], data['y_tr'], data['X_va'],
                                        data['y_va'], data['X_te'], data['y_te'])
        nf, nc = data['n_features'], data['n_classes']
        pe, *_ = train_teacher_ensemble(data, TEACHER_CONFIGS['base'], seed)
        s_tr = pe(Xtr)
        for t in targets:
            u, l = MLP_CONFIGS[t]
            tf.random.set_seed(seed)
            mb = create_small_mlp(u, l, nf, nc)
            mb.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            mb.fit(Xtr.astype(np.float32), ytr.astype(np.int64), epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            base_acc = accuracy_score(yte, mb.predict(Xte, verbose=0).argmax(1))
            bv, bt, best_pred = 0, 0, None
            for a in [0.0, 0.3, 0.5, 0.7]:
                for T in TEMPERATURES:
                    v, te, probs = train_mlp_distilled(Xtr, ytr, s_tr, Xva, yva, Xte, yte, u, l, nf, nc, a, T, seed)
                    if v > bv:
                        bv, bt, best_pred = v, te, probs.argmax(1)
            per_seed[t].append((bt - base_acc) * 100)
            if t == 'MLP(8)':
                bpred = mb.predict(Xte, verbose=0).argmax(1)
                bcorr = (bpred == yte); dcorr = (best_pred == yte)
                mcnemar['b_only'] += int(np.sum(bcorr & ~dcorr))
                mcnemar['d_only'] += int(np.sum(~bcorr & dcorr))
                n_test_total += len(yte)
        print(f"({time.time()-t0:.0f}s)", end="")
    print()
    print(f"\n{'Model':<10} {'mean pp':<9} {'sd':<7} {'dz':<7} {'CI width':<10} {'power':<8} {'+correct/seed':<14}")
    print("-" * 70)
    summ = {}
    rng = np.random.RandomState(0)
    test_n = len(yte)
    for t in targets:
        x = np.array(per_seed[t]); n = len(x); dz = cohens_dz(x)
        ci = stats.t.interval(0.95, n - 1, loc=x.mean(), scale=stats.sem(x)); ciw = ci[1] - ci[0]
        from math import sqrt
        ncp = dz * sqrt(n); tcrit = stats.t.ppf(0.95, n - 1)
        power = float(1 - stats.nct.cdf(tcrit, n - 1, ncp))
        add_correct = x.mean() / 100 * test_n
        summ[t] = {'mean_pp': float(x.mean()), 'sd': float(x.std(ddof=1)), 'dz': dz,
                   'ci95': [float(ci[0]), float(ci[1])], 'ci_width': float(ciw),
                   'power': power, 'added_correct_per_seed': float(add_correct),
                   'per_seed': x.tolist()}
        print(f"{t:<10} {x.mean():<9.2f} {x.std(ddof=1):<7.2f} {dz:<7.2f} {ciw:<10.2f} "
              f"{power:<8.3f} {add_correct:<14.1f}")
    # McNemar exact test on pooled discordant pairs
    b, d = mcnemar['b_only'], mcnemar['d_only']
    p_mcnemar = float(stats.binomtest(min(b, d), b + d, 0.5).pvalue) if (b + d) > 0 else 1.0
    print(f"\n  McNemar (pooled, MLP(8)): baseline-only correct={b}, distilled-only correct={d}, "
          f"p={p_mcnemar:.2e}")
    print(f"  => '+0.5pp' is small in relative terms but reflects ~{summ['MLP(8)']['added_correct_per_seed']:.0f} "
          f"extra correct test predictions per seed out of {test_n}.")
    DUMP['C11'] = {'effect_sizes': summ,
                   'mcnemar_MLP8': {'baseline_only': b, 'distilled_only': d, 'p_value': p_mcnemar}}
    return summ


# ============================================================
# MAIN
# ============================================================
EXPERIMENTS = {
    'C1': exp_C1, 'C2': exp_C2, 'C3': exp_C3, 'C4': exp_C4, 'C5': exp_C5, 'C6': exp_C6,
    'C7': exp_C7, 'C8': exp_C8, 'C9': exp_C9, 'C10': exp_C10, 'C11': exp_C11,
}

if __name__ == '__main__':
    t_start = time.time()
    print("=" * 80)
    print("CLASSIFICATION KD — REVIEWER-RESPONSE EXPERIMENTS")
    print(f"Mode: {MODE} | Seeds: {N_SEEDS} | Only: {ONLY or 'all'}")
    print("=" * 80)

    for key, fn in EXPERIMENTS.items():
        if ONLY and key not in ONLY:
            continue
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {key} failed: {e}")
            traceback.print_exc()
            DUMP.setdefault('_errors', {})[key] = str(e)

    def _native(o):
        if isinstance(o, dict):
            return {(k if isinstance(k, str) else str(k)): _native(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_native(v) for v in o]
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.bool_): return bool(o)
        return o

    # Merge into any existing results file so `--only X` augments rather than clobbers.
    out_path = 'results_classification_revision_full.json'
    merged = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                merged = json.load(f)
        except Exception:
            merged = {}
    merged.update(_native(DUMP))
    with open(out_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"\nWrote {out_path} (merged; keys: {sorted(k for k in merged if not k.startswith('_'))})")
    print(f"\n{'=' * 80}\nDONE in {(time.time()-t_start)/60:.1f} min\n{'=' * 80}")