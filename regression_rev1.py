"""
Reviewer-response experiments for the REGRESSION knowledge-distillation study.
=============================================================================
This companion script ADDS the experiments requested by the two reviewers.
It imports the building blocks from the submitted `regression.py` (data
generator, teacher/student builders, aggregation), so the original script
stays untouched and continues to reproduce the submitted tables.

Place this file in the SAME directory as regression.py and run:
  python regression_revision.py --quick   # 3 seeds, tiny grids (smoke test)
  python regression_revision.py           # 10 seeds
  python regression_revision.py --full    # 20 seeds (paper run)
  python regression_revision.py --xl      # 30 seeds (seed-sufficiency check)

You can also restrict to a subset of experiments, e.g.:
  python regression_revision.py --full --only R1,R2,R7

Experiments (each maps to reviewer comments; see EXPERIMENTS dict):
  R1  Teacher quality controls: oracle / single / homogeneous / heterogeneous /
      shuffled teacher                                  (R1.1, R1.8, R2.7)
  R2  Epistemic vs aleatoric uncertainty decomposition  (R1.2)
  R3  Harder data conditions (correlated / heavy-tailed /
      heteroscedastic / non-Gaussian)                   (R1.3, R2.2)
  R4  Robustness: covariate shift + input perturbation  (R1.11)
  R5  Capacity ladder (gain vs student capacity)        (R1.12)
  R6  Computational complexity / cost-benefit           (R1.10)
  R7  Seed sufficiency & retrospective power            (R1.7)
  R8  Extra baselines: mixup / dropout / born-again /
      input-noise                                        (R2.4)
  R9  Additional real datasets (OpenML, network-guarded) (R1.14)

All numeric output is also written to:
  results_regression_revision_full.json
  results_regression_revision_*.csv
"""

import os, sys, time, json, csv, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---- reuse the submitted pipeline -----------------------------------------
from regression import (
    N_SAMPLES, N_FEATURES, BASE_SIGMA, ALPHAS, MLP_EPOCHS, MLP_BATCH,
    TEACHER_CONFIGS, MLP_CONFIGS,
    true_function, generate_synthetic,
    create_teacher_nn, create_small_mlp, get_sklearn_students,
    train_teacher_ensemble, aggregate_seeds, print_table,
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
        i = sys.argv.index('--only')
        return set(s.strip().upper() for s in sys.argv[i + 1].split(','))
    return None
ONLY = _only()

DUMP = {'config': {'mode': MODE, 'n_seeds': N_SEEDS, 'seeds': SEEDS}}


# ============================================================
# SHARED HELPERS
# ============================================================
def distill_students(data, seed, t_train, t_test, alphas=ALPHAS,
                     include_mlp=True, include_sklearn=True):
    """Train every student on alpha-blended targets. Returns results dict
    {model: {alpha: {'val','test'}}} compatible with aggregate_seeds()."""
    X_tr, X_va, X_te = data['X_tr'], data['X_va'], data['X_te']
    y_tr, y_va, y_te = data['y_tr'], data['y_va'], data['y_te']
    nf = data['n_features']
    results, timing = {}, {}

    if include_sklearn:
        for name, model in get_sklearn_students(seed).items():
            results[name] = {}
            t0 = time.time()
            for a in alphas:
                m = clone(model)
                blended = a * y_tr + (1 - a) * t_train
                m.fit(X_tr, blended)
                results[name][a] = {
                    'val': mean_squared_error(y_va, m.predict(X_va)),
                    'test': mean_squared_error(y_te, m.predict(X_te))}
            timing[name] = time.time() - t0

    if include_mlp:
        for name, (units, nl) in MLP_CONFIGS.items():
            results[name] = {}
            t0 = time.time()
            for a in alphas:
                tf.random.set_seed(seed); np.random.seed(seed)
                m = create_small_mlp(units, nl, nf)
                blended = a * y_tr + (1 - a) * t_train
                m.compile(optimizer='adam', loss='mse')
                m.fit(X_tr, blended, epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
                results[name][a] = {
                    'val': mean_squared_error(y_va, m.predict(X_va, verbose=0).flatten()),
                    'test': mean_squared_error(y_te, m.predict(X_te, verbose=0).flatten())}
            timing[name] = time.time() - t0

    return results, timing


def make_meta(data, t_train, t_test, members_train=None,
              teacher_time=0.0, student_timing=None):
    """Build a meta dict compatible with aggregate_seeds(), plus extra
    teacher diagnostics (bias to f*, epistemic disagreement)."""
    y_tr, y_te = data['y_tr'], data['y_te']
    yt_tr = data.get('yt_tr')
    teacher_mse = mean_squared_error(y_te, t_test)
    noise_var = tvt = den = None
    if yt_tr is not None:
        noise_var = mean_squared_error(yt_tr, y_tr)          # E[(y - f*)^2] = sigma^2
        tvt = mean_squared_error(yt_tr, t_train)             # teacher error vs conditional mean
        den = (1 - tvt / noise_var) * 100 if noise_var > 0 else 0
    epistemic = None
    if members_train is not None and len(members_train) > 1:
        M = np.stack(members_train, axis=0)                  # (n_members, n_samples)
        epistemic = float(np.mean(np.var(M, axis=0)))        # mean across-member variance
    return {
        'teacher_mse': teacher_mse, 'noise_var': noise_var,
        'teacher_vs_true': tvt, 'denoising_pct': den,
        'epistemic_var': epistemic,
        'teacher_time': teacher_time,
        'student_timing': student_timing or {},
    }


def cohens_dz(delta):
    """Paired effect size: mean(diff)/sd(diff)."""
    delta = np.asarray(delta, float)
    sd = delta.std(ddof=1)
    return float(delta.mean() / sd) if sd > 0 else 0.0


def summarise(agg):
    """Compact dict of family-level numbers from an aggregate_seeds() output."""
    models = {k: v for k, v in agg.items() if not k.startswith('_')}
    helped = sum(1 for v in models.values() if v['imp_mean'] > 0)
    avg = float(np.mean([v['imp_mean'] for v in models.values()]))
    mlp = float(np.mean([v['imp_mean'] for k, v in models.items() if 'MLP' in k]))
    return {'helped': helped, 'n_models': len(models),
            'avg_improv_pct': avg, 'mlp_improv_pct': mlp}


# ============================================================
# TEACHER CONSTRUCTORS (for the teacher-quality experiment)
# ============================================================
def build_teacher(data, seed, kind, cfg_name='base'):
    """Return (t_train, t_test, members_train, info).
       members_train: list of per-member predictions on X_tr (for epistemic)."""
    X_tr, X_te = data['X_tr'], data['X_te']
    y_tr = data['y_tr']
    cfg = TEACHER_CONFIGS[cfg_name]

    if kind == 'oracle':                                     # upper bound on teacher ability
        return data['yt_tr'], data['yt_te'], [data['yt_tr']], {'note': 'oracle = f*'}

    if kind in ('hetero', 'nn', 'xgb', 'gb'):
        pe, pn, px, pg = train_teacher_ensemble(data, cfg, seed)
        fn = {'hetero': pe, 'nn': pn, 'xgb': px, 'gb': pg}[kind]
        members = [pn(X_tr), px(X_tr), pg(X_tr)] if kind == 'hetero' else [fn(X_tr)]
        return fn(X_tr), fn(X_te), members, {}

    if kind == 'shuffled':                                   # negative control: kill per-sample ability
        pe, _, _, _ = train_teacher_ensemble(data, cfg, seed)
        ttr = pe(X_tr)
        rng = np.random.RandomState(seed + 777)
        return ttr[rng.permutation(len(ttr))], pe(X_te), [ttr], {'note': 'shuffled control'}

    if kind.startswith('homo_'):                             # homogeneous bootstrap ensemble (3 members)
        base = kind.split('_', 1)[1]
        ptr, pte = [], []
        for j in range(3):
            sj = seed + 1000 * (j + 1)
            if base == 'nn':
                tf.random.set_seed(sj)
                nn = create_teacher_nn(cfg['nn_layers'], data['n_features'])
                nn.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
                nn.fit(X_tr, y_tr, epochs=cfg['nn_epochs'], batch_size=32,
                       validation_data=(data['X_va'], data['y_va']),
                       callbacks=[keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
                                  keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)],
                       verbose=0)
                ptr.append(nn.predict(X_tr, verbose=0).flatten())
                pte.append(nn.predict(X_te, verbose=0).flatten())
            elif base == 'xgb':
                m = XGBRegressor(n_estimators=cfg['xgb_n'], max_depth=cfg['xgb_d'],
                                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                 random_state=sj, verbosity=0)
                m.fit(X_tr, y_tr); ptr.append(m.predict(X_tr)); pte.append(m.predict(X_te))
            elif base == 'gb':
                m = GradientBoostingRegressor(n_estimators=cfg['gb_n'], max_depth=cfg['gb_d'],
                                              learning_rate=0.05, subsample=0.8, random_state=sj)
                m.fit(X_tr, y_tr); ptr.append(m.predict(X_tr)); pte.append(m.predict(X_te))
        return np.mean(ptr, axis=0), np.mean(pte, axis=0), ptr, {}

    raise ValueError(kind)


# ============================================================
# R1: TEACHER-QUALITY CONTROLS   (R1.1 ability-vs-variance, R1.8 homo/hetero, R2.7 -> E[y|x])
# ============================================================
def exp_R1():
    print("\n" + "=" * 80)
    print(f"R1  TEACHER-QUALITY CONTROLS (seeds={N_SEEDS})")
    print("    oracle / single / homogeneous / heterogeneous / shuffled")
    print("=" * 80)
    kinds = ['oracle', 'hetero', 'nn', 'xgb', 'gb',
             'homo_nn', 'homo_xgb', 'homo_gb', 'shuffled']
    if MODE == 'QUICK':
        kinds = ['oracle', 'hetero', 'gb', 'homo_gb', 'shuffled']

    out = {}
    for kind in kinds:
        print(f"\n--- teacher = {kind} ---")
        all_data, t_to_fstar, epi = [], [], []
        for s, seed in enumerate(SEEDS):
            print(f"  seed {s+1}/{N_SEEDS}", end=" ", flush=True)
            t0 = time.time()
            data = generate_synthetic(seed, BASE_SIGMA)
            t_tr, t_te, members, _ = build_teacher(data, seed, kind)
            res, tim = distill_students(data, seed, t_tr, t_te)
            meta = make_meta(data, t_tr, t_te, members, student_timing=tim)
            all_data.append((res, meta))
            if meta['teacher_vs_true'] is not None:
                t_to_fstar.append(meta['teacher_vs_true'])
            if meta['epistemic_var'] is not None:
                epi.append(meta['epistemic_var'])
            print(f"({time.time()-t0:.0f}s)", end="")
        print()
        agg = aggregate_seeds(all_data)
        s = summarise(agg)
        s['teacher_mse'] = float(agg['_teacher']['mse_mean'])
        s['teacher_mse_to_fstar'] = float(np.mean(t_to_fstar)) if t_to_fstar else None
        s['denoise_pct'] = float(agg['_teacher']['denoise']) if agg['_teacher']['denoise'] is not None else None
        s['epistemic_var'] = float(np.mean(epi)) if epi else None
        out[kind] = {'summary': s, 'aggregate': agg}
        print(f"  teacher MSE(vs y)={s['teacher_mse']:.3f}  "
              f"MSE(vs f*)={s['teacher_mse_to_fstar']}  "
              f"helped={s['helped']}/{s['n_models']}  "
              f"avg={s['avg_improv_pct']:+.2f}%  MLP={s['mlp_improv_pct']:+.2f}%")

    print("\n" + "-" * 80)
    print(f"{'Teacher':<12} {'MSE(y)':<9} {'MSE(f*)':<9} {'Denoise%':<9} "
          f"{'Epist.var':<10} {'Helped':<8} {'Avg%':<8} {'MLP%':<8}")
    print("-" * 80)
    for kind in kinds:
        s = out[kind]['summary']
        mf = f"{s['teacher_mse_to_fstar']:.3f}" if s['teacher_mse_to_fstar'] is not None else "  -  "
        dn = f"{s['denoise_pct']:.1f}" if s['denoise_pct'] is not None else " - "
        ev = f"{s['epistemic_var']:.4f}" if s['epistemic_var'] is not None else "  -  "
        print(f"{kind:<12} {s['teacher_mse']:<9.3f} {mf:<9} {dn:<9} {ev:<10} "
              f"{s['helped']}/{s['n_models']:<6} {s['avg_improv_pct']:<+8.2f} {s['mlp_improv_pct']:<+8.2f}")
    print("\nReading: oracle = ability ceiling; homo_* isolate variance reduction within one "
          "hypothesis class; shuffled keeps the teacher's marginal but destroys per-sample "
          "ability. If gains track MSE(vs f*) and collapse for 'shuffled', the benefit is "
          "teacher ABILITY, not mere ensemble variance reduction.")
    DUMP['R1'] = {k: out[k]['summary'] for k in out}
    DUMP['R1_full'] = {k: out[k]['aggregate'] for k in out}
    return out


# ============================================================
# R2: EPISTEMIC vs ALEATORIC UNCERTAINTY   (R1.2)
# ============================================================
def exp_R2():
    print("\n" + "=" * 80)
    print(f"R2  UNCERTAINTY DECOMPOSITION (seeds={N_SEEDS})")
    print("=" * 80)
    rows = []
    for s, seed in enumerate(SEEDS):
        print(f"  seed {s+1}/{N_SEEDS}", end=" ", flush=True)
        t0 = time.time()
        data = generate_synthetic(seed, BASE_SIGMA)
        pe, pn, px, pg = train_teacher_ensemble(data, TEACHER_CONFIGS['base'], seed)
        Xtr = data['X_tr']
        members = np.stack([pn(Xtr), px(Xtr), pg(Xtr)], axis=0)   # (3, n)
        epistemic = np.var(members, axis=0)                       # per-sample disagreement
        aleatoric = BASE_SIGMA ** 2                               # known irreducible noise
        ens = members.mean(axis=0)
        err_to_fstar = np.abs(ens - data['yt_tr'])                # |teacher - E[y|x]|
        rows.append({
            'seed': seed,
            'epistemic_mean': float(epistemic.mean()),
            'aleatoric': float(aleatoric),
            'corr_epi_err': float(np.corrcoef(epistemic, err_to_fstar)[0, 1]),
            'epi_over_total': float(epistemic.mean() / (epistemic.mean() + aleatoric)),
        })
        print(f"({time.time()-t0:.0f}s)", end="")
    print()
    epi = np.mean([r['epistemic_mean'] for r in rows])
    corr = np.mean([r['corr_epi_err'] for r in rows])
    frac = np.mean([r['epi_over_total'] for r in rows])
    print(f"\n  mean epistemic var          : {epi:.4f}")
    print(f"  aleatoric var (= sigma^2)   : {BASE_SIGMA**2:.4f}")
    print(f"  epistemic / (epi+aleatoric) : {frac:.3f}")
    print(f"  corr(epistemic, |teacher-f*|): {corr:+.3f}  "
          "(positive => disagreement flags where the teacher is least reliable)")
    DUMP['R2'] = {'per_seed': rows,
                  'epistemic_mean': float(epi), 'aleatoric': BASE_SIGMA**2,
                  'epi_fraction': float(frac), 'corr_epi_err': float(corr)}
    return rows


# ============================================================
# R3: HARDER DATA CONDITIONS   (R1.3, R2.2)
# ============================================================
def _split(X, y, y_true, seed):
    X_tr, X_te, y_tr, y_te, yt_tr, yt_te = train_test_split(
        X, y, y_true, test_size=0.2, random_state=seed)
    X_tr, X_va, y_tr, y_va, yt_tr, yt_va = train_test_split(
        X_tr, y_tr, yt_tr, test_size=0.15, random_state=seed)
    return dict(X_tr=X_tr, X_va=X_va, X_te=X_te, y_tr=y_tr, y_va=y_va, y_te=y_te,
                yt_tr=yt_tr, yt_te=yt_te, n_features=X.shape[1])


def gen_condition(seed, condition):
    """Synthetic regression data with a harder generating process."""
    rng = np.random.RandomState(seed)
    n, d = N_SAMPLES, N_FEATURES
    if condition == 'correlated':
        cov = np.full((d, d), 0.7) + 0.3 * np.eye(d)
        X = rng.multivariate_normal(np.zeros(d), cov, size=n)
        y_true = true_function(X); y = y_true + BASE_SIGMA * rng.randn(n)
    elif condition == 'heavy_tailed':                         # Student-t(3) noise
        X = rng.randn(n, d); y_true = true_function(X)
        y = y_true + BASE_SIGMA * rng.standard_t(df=3, size=n)
    elif condition == 'heteroscedastic':                      # noise scales with |x1|
        X = rng.randn(n, d); y_true = true_function(X)
        scale = BASE_SIGMA * (0.5 + np.abs(X[:, 0]))
        y = y_true + scale * rng.randn(n)
    elif condition == 'non_gaussian':                         # skewed exponential features
        X = rng.exponential(1.0, size=(n, d)) - 1.0
        y_true = true_function(X); y = y_true + BASE_SIGMA * rng.randn(n)
    else:
        raise ValueError(condition)
    return _split(X, y, y_true, seed)


def exp_R3():
    print("\n" + "=" * 80)
    print(f"R3  HARDER DATA CONDITIONS (seeds={N_SEEDS})")
    print("=" * 80)
    conds = ['correlated', 'heavy_tailed', 'heteroscedastic', 'non_gaussian']
    if MODE == 'QUICK':
        conds = ['correlated', 'heavy_tailed']
    out = {}
    for cond in conds:
        print(f"\n--- condition = {cond} ---")
        all_data = []
        for s, seed in enumerate(SEEDS):
            print(f"  seed {s+1}/{N_SEEDS}", end=" ", flush=True)
            t0 = time.time()
            data = gen_condition(seed, cond)
            pe, pn, px, pg = train_teacher_ensemble(data, TEACHER_CONFIGS['base'], seed)
            t_tr, t_te = pe(data['X_tr']), pe(data['X_te'])
            res, tim = distill_students(data, seed, t_tr, t_te)
            meta = make_meta(data, t_tr, t_te, [pn(data['X_tr']), px(data['X_tr']), pg(data['X_tr'])],
                             student_timing=tim)
            all_data.append((res, meta))
            print(f"({time.time()-t0:.0f}s)", end="")
        print()
        agg = aggregate_seeds(all_data)
        out[cond] = {'summary': summarise(agg), 'aggregate': agg,
                     'teacher_mse': float(agg['_teacher']['mse_mean']),
                     'denoise_pct': agg['_teacher']['denoise']}
        sm = out[cond]['summary']
        print(f"  teacher MSE={out[cond]['teacher_mse']:.3f}  helped={sm['helped']}/{sm['n_models']}  "
              f"avg={sm['avg_improv_pct']:+.2f}%  MLP={sm['mlp_improv_pct']:+.2f}%")
    DUMP['R3'] = {c: out[c]['summary'] | {'teacher_mse': out[c]['teacher_mse'],
                                          'denoise_pct': out[c]['denoise_pct']} for c in out}
    DUMP['R3_full'] = {c: out[c]['aggregate'] for c in out}
    return out


# ============================================================
# R4: ROBUSTNESS — covariate shift + input perturbation   (R1.11)
# ============================================================
def exp_R4():
    print("\n" + "=" * 80)
    print(f"R4  ROBUSTNESS: covariate shift + input perturbation (seeds={N_SEEDS})")
    print("=" * 80)
    shifts = {'clean': None, 'mean_shift': 0.5, 'scale_shift': 1.5, 'gauss_pert': 0.2}
    # baseline (alpha=1) vs distilled (best alpha=0.3) for representative MLPs
    targets = ['MLP(8x2)', 'MLP(8)', 'MLP(16x2)']
    rows = {t: {sh: {'base': [], 'distil': []} for sh in shifts} for t in targets}

    for s, seed in enumerate(SEEDS):
        print(f"  seed {s+1}/{N_SEEDS}", end=" ", flush=True)
        t0 = time.time()
        data = generate_synthetic(seed, BASE_SIGMA)
        pe, *_ = train_teacher_ensemble(data, TEACHER_CONFIGS['base'], seed)
        t_tr = pe(data['X_tr'])
        Xtr, ytr = data['X_tr'], data['y_tr']
        Xte, yte = data['X_te'], data['y_te']
        for t in targets:
            units, nl = MLP_CONFIGS[t]
            for tag, alpha in [('base', 1.0), ('distil', 0.3)]:
                tf.random.set_seed(seed); np.random.seed(seed)
                m = create_small_mlp(units, nl, data['n_features'])
                blended = alpha * ytr + (1 - alpha) * t_tr
                m.compile(optimizer='adam', loss='mse')
                m.fit(Xtr, blended, epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
                for sh, mag in shifts.items():
                    if sh == 'clean':
                        Xev = Xte
                    elif sh == 'mean_shift':
                        Xev = Xte + mag
                    elif sh == 'scale_shift':
                        Xev = Xte * mag
                    elif sh == 'gauss_pert':
                        Xev = Xte + mag * np.random.RandomState(seed + 5).randn(*Xte.shape)
                    mse = mean_squared_error(yte, m.predict(Xev, verbose=0).flatten())
                    rows[t][sh][tag].append(mse)
        print(f"({time.time()-t0:.0f}s)", end="")
    print()

    print(f"\n{'Model':<10} {'Shift':<12} {'Base MSE':<12} {'Distil MSE':<12} {'Distil better?':<14}")
    print("-" * 64)
    summ = {}
    for t in targets:
        summ[t] = {}
        for sh in shifts:
            b = float(np.mean(rows[t][sh]['base']))
            d = float(np.mean(rows[t][sh]['distil']))
            summ[t][sh] = {'base': b, 'distil': d, 'distil_better': bool(d < b)}
            print(f"{t:<10} {sh:<12} {b:<12.3f} {d:<12.3f} {'yes' if d < b else 'no':<14}")
    DUMP['R4'] = summ
    return summ


# ============================================================
# R5: CAPACITY LADDER   (R1.12)
# ============================================================
LADDER = [(2, 1), (4, 1), (8, 1), (16, 1), (32, 1), (64, 1), (8, 2), (16, 2), (32, 2)]

def exp_R5():
    print("\n" + "=" * 80)
    print(f"R5  CAPACITY LADDER: gain vs student capacity (seeds={N_SEEDS})")
    print("=" * 80)
    ladder = LADDER if MODE != 'QUICK' else [(4, 1), (16, 1), (32, 2)]
    acc = {f"MLP({u}x{l})" if l > 1 else f"MLP({u})": {'base': [], 'distil': [], 'params': None}
           for (u, l) in ladder}
    names = list(acc.keys())

    for s, seed in enumerate(SEEDS):
        print(f"  seed {s+1}/{N_SEEDS}", end=" ", flush=True)
        t0 = time.time()
        data = generate_synthetic(seed, BASE_SIGMA)
        pe, *_ = train_teacher_ensemble(data, TEACHER_CONFIGS['base'], seed)
        t_tr = pe(data['X_tr'])
        for (u, l), name in zip(ladder, names):
            best_val, best_test, base_test = np.inf, None, None
            for a in [0.0, 0.3, 0.5, 0.7, 1.0]:
                tf.random.set_seed(seed); np.random.seed(seed)
                m = create_small_mlp(u, l, data['n_features'])
                if acc[name]['params'] is None:
                    acc[name]['params'] = int(m.count_params())
                blended = a * data['y_tr'] + (1 - a) * t_tr
                m.compile(optimizer='adam', loss='mse')
                m.fit(data['X_tr'], blended, epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
                v = mean_squared_error(data['y_va'], m.predict(data['X_va'], verbose=0).flatten())
                te = mean_squared_error(data['y_te'], m.predict(data['X_te'], verbose=0).flatten())
                if a == 1.0:
                    base_test = te
                if v < best_val:
                    best_val, best_test = v, te
            acc[name]['base'].append(base_test)
            acc[name]['distil'].append(best_test)
        print(f"({time.time()-t0:.0f}s)", end="")
    print()

    print(f"\n{'Model':<10} {'Params':<8} {'Base MSE':<11} {'Distil MSE':<12} {'Gain%':<8}")
    print("-" * 52)
    summ = {}
    for name in names:
        b = np.array(acc[name]['base']); d = np.array(acc[name]['distil'])
        gain = float(((b - d) / b * 100).mean())
        summ[name] = {'params': acc[name]['params'], 'base_mse': float(b.mean()),
                      'distil_mse': float(d.mean()), 'gain_pct': gain}
        print(f"{name:<10} {acc[name]['params']:<8} {b.mean():<11.3f} {d.mean():<12.3f} {gain:<+8.2f}")
    DUMP['R5'] = summ
    return summ


# ============================================================
# R6: COMPUTATIONAL COMPLEXITY / COST-BENEFIT   (R1.10)
# ============================================================
def exp_R6():
    print("\n" + "=" * 80)
    print(f"R6  COMPUTATIONAL COST-BENEFIT (seeds={min(N_SEEDS,5)})")
    print("=" * 80)
    n = min(N_SEEDS, 5)
    teacher_times, sweep_times, infer = [], [], {}
    params = {}
    for s in range(n):
        seed = 42 + s
        print(f"  seed {s+1}/{n}", end=" ", flush=True)
        data = generate_synthetic(seed, BASE_SIGMA)
        t0 = time.time()
        pe, *_ = train_teacher_ensemble(data, TEACHER_CONFIGS['base'], seed)
        teacher_times.append(time.time() - t0)
        t_tr = pe(data['X_tr'])
        for name, (u, l) in MLP_CONFIGS.items():
            t0 = time.time()
            for a in ALPHAS:                                  # full alpha sweep cost
                tf.random.set_seed(seed)
                m = create_small_mlp(u, l, data['n_features'])
                blended = a * data['y_tr'] + (1 - a) * t_tr
                m.compile(optimizer='adam', loss='mse')
                m.fit(data['X_tr'], blended, epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            sweep_times.append((name, time.time() - t0))
            params.setdefault(name, int(m.count_params()))
            t1 = time.time()
            for _ in range(5):
                m.predict(data['X_te'], verbose=0)
            infer.setdefault(name, []).append((time.time() - t1) / 5)
        print("done")
    summ = {
        'teacher_train_s': float(np.mean(teacher_times)),
        'alpha_sweep_s_by_model': {nm: float(np.mean([t for (n_, t) in sweep_times if n_ == nm]))
                                   for nm in MLP_CONFIGS},
        'params': params,
        'inference_s': {nm: float(np.mean(v)) for nm, v in infer.items()},
    }
    print(f"\n  teacher ensemble train: {summ['teacher_train_s']:.1f}s (mean)")
    print(f"  {'Model':<10} {'Params':<8} {'alpha-sweep s':<14} {'infer s':<10}")
    for nm in MLP_CONFIGS:
        print(f"  {nm:<10} {summ['params'][nm]:<8} "
              f"{summ['alpha_sweep_s_by_model'][nm]:<14.2f} {summ['inference_s'][nm]:<10.4f}")
    print("  (cost is amortised: the teacher + sweep are paid once; the deployed student is "
          "the same size as the baseline, so inference cost is unchanged.)")
    DUMP['R6'] = summ
    return summ


# ============================================================
# R7: SEED SUFFICIENCY & RETROSPECTIVE POWER   (R1.7)
# ============================================================
def exp_R7():
    print("\n" + "=" * 80)
    print(f"R7  SEED SUFFICIENCY & POWER (running on {N_SEEDS} seeds)")
    print("=" * 80)
    # Run the main distillation and collect per-seed MLP deltas (improvement %).
    targets = ['MLP(8x2)', 'MLP(8)', 'MLP(16)', 'MLP(4)', 'MLP(16x2)']
    per_seed = {t: [] for t in targets}
    for s, seed in enumerate(SEEDS):
        print(f"  seed {s+1}/{N_SEEDS}", end=" ", flush=True)
        t0 = time.time()
        data = generate_synthetic(seed, BASE_SIGMA)
        pe, *_ = train_teacher_ensemble(data, TEACHER_CONFIGS['base'], seed)
        t_tr = pe(data['X_tr'])
        res, _ = distill_students(data, seed, t_tr, pe(data['X_te']),
                                  include_sklearn=False)
        for t in targets:
            r = res[t]
            base = r[1.0]['test']
            best_a = min(ALPHAS, key=lambda a: r[a]['val'])
            imp = (base - r[best_a]['test']) / base * 100
            per_seed[t].append(imp)
        print(f"({time.time()-t0:.0f}s)", end="")
    print()

    print(f"\n{'Model':<10} {'mean%':<8} {'sd':<8} {'dz':<7} {'CI width@n':<12} {'power(a=.05)':<12}")
    print("-" * 60)
    summ = {}
    rng = np.random.RandomState(0)
    for t in targets:
        x = np.array(per_seed[t]); n = len(x)
        dz = cohens_dz(x)
        ci = stats.t.interval(0.95, n - 1, loc=x.mean(), scale=stats.sem(x))
        ciw = ci[1] - ci[0]
        # retrospective power for a one-sided paired t at observed effect
        from math import sqrt
        ncp = dz * sqrt(n)
        tcrit = stats.t.ppf(0.95, n - 1)
        power = float(1 - stats.nct.cdf(tcrit, n - 1, ncp))
        # bootstrap stability of the mean across seeds
        boot = [rng.choice(x, n, replace=True).mean() for _ in range(2000)]
        summ[t] = {'mean_pct': float(x.mean()), 'sd': float(x.std(ddof=1)),
                   'dz': dz, 'ci95': [float(ci[0]), float(ci[1])],
                   'ci_width': float(ciw), 'power': power,
                   'bootstrap_sd': float(np.std(boot)),
                   'per_seed': x.tolist()}
        print(f"{t:<10} {x.mean():<8.2f} {x.std(ddof=1):<8.2f} {dz:<7.2f} "
              f"{ciw:<12.2f} {power:<12.3f}")
    print("\n  Running-mean convergence (cumulative mean improvement %, MLP(8x2)):")
    x = np.array(per_seed['MLP(8x2)'])
    run = np.cumsum(x) / np.arange(1, len(x) + 1)
    print("   " + "  ".join(f"n={i+1}:{run[i]:.1f}" for i in range(0, len(run), max(1, len(run)//8))))
    DUMP['R7'] = summ
    return summ


# ============================================================
# R8: EXTRA BASELINES — mixup / dropout / born-again / input-noise   (R2.4)
# ============================================================
def _mlp_with_dropout(u, l, nf, rate=0.2):
    ll = [layers.Dense(u, activation='relu', input_shape=(nf,)), layers.Dropout(rate)]
    for _ in range(l - 1):
        ll += [layers.Dense(u, activation='relu'), layers.Dropout(rate)]
    ll.append(layers.Dense(1))
    return keras.Sequential(ll)


def exp_R8():
    print("\n" + "=" * 80)
    print(f"R8  EXTRA BASELINES vs KD for MLPs (seeds={N_SEEDS})")
    print("    standard / mixup / dropout / input-noise / born-again / KD")
    print("=" * 80)
    res = {name: {k: [] for k in ['standard', 'mixup', 'dropout',
                                  'input_noise', 'born_again', 'KD']}
           for name in MLP_CONFIGS}
    for s, seed in enumerate(SEEDS):
        print(f"  seed {s+1}/{N_SEEDS}", end=" ", flush=True)
        t0 = time.time()
        data = generate_synthetic(seed, BASE_SIGMA)
        Xtr, ytr, Xva, yva, Xte, yte = (data['X_tr'], data['y_tr'], data['X_va'],
                                        data['y_va'], data['X_te'], data['y_te'])
        nf = data['n_features']
        pe, *_ = train_teacher_ensemble(data, TEACHER_CONFIGS['base'], seed)
        t_tr = pe(Xtr)
        rng = np.random.RandomState(seed)
        for name, (u, l) in MLP_CONFIGS.items():
            # standard
            tf.random.set_seed(seed)
            m = create_small_mlp(u, l, nf); m.compile(optimizer='adam', loss='mse')
            m.fit(Xtr, ytr, epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            res[name]['standard'].append(mean_squared_error(yte, m.predict(Xte, verbose=0).flatten()))
            # mixup (lambda~Beta(0.2,0.2))
            tf.random.set_seed(seed)
            lam = rng.beta(0.2, 0.2, size=len(Xtr))[:, None]
            perm = rng.permutation(len(Xtr))
            Xmix = lam * Xtr + (1 - lam) * Xtr[perm]
            ymix = lam[:, 0] * ytr + (1 - lam[:, 0]) * ytr[perm]
            m = create_small_mlp(u, l, nf); m.compile(optimizer='adam', loss='mse')
            m.fit(Xmix, ymix, epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            res[name]['mixup'].append(mean_squared_error(yte, m.predict(Xte, verbose=0).flatten()))
            # dropout
            tf.random.set_seed(seed)
            m = _mlp_with_dropout(u, l, nf); m.compile(optimizer='adam', loss='mse')
            m.fit(Xtr, ytr, epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            res[name]['dropout'].append(mean_squared_error(yte, m.predict(Xte, verbose=0).flatten()))
            # input noise
            tf.random.set_seed(seed)
            Xn = Xtr + 0.1 * rng.randn(*Xtr.shape)
            m = create_small_mlp(u, l, nf); m.compile(optimizer='adam', loss='mse')
            m.fit(Xn, ytr, epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            res[name]['input_noise'].append(mean_squared_error(yte, m.predict(Xte, verbose=0).flatten()))
            # born-again self-distillation (student teaches identical student)
            tf.random.set_seed(seed)
            m1 = create_small_mlp(u, l, nf); m1.compile(optimizer='adam', loss='mse')
            m1.fit(Xtr, ytr, epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            ba_target = 0.5 * ytr + 0.5 * m1.predict(Xtr, verbose=0).flatten()
            tf.random.set_seed(seed)
            m2 = create_small_mlp(u, l, nf); m2.compile(optimizer='adam', loss='mse')
            m2.fit(Xtr, ba_target, epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            res[name]['born_again'].append(mean_squared_error(yte, m2.predict(Xte, verbose=0).flatten()))
            # KD (validation-selected alpha)
            best_val, best_test = np.inf, None
            for a in [0.0, 0.3, 0.5, 0.7]:
                tf.random.set_seed(seed)
                m = create_small_mlp(u, l, nf); m.compile(optimizer='adam', loss='mse')
                m.fit(Xtr, a * ytr + (1 - a) * t_tr, epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
                v = mean_squared_error(yva, m.predict(Xva, verbose=0).flatten())
                te = mean_squared_error(yte, m.predict(Xte, verbose=0).flatten())
                if v < best_val:
                    best_val, best_test = v, te
            res[name]['KD'].append(best_test)
        print(f"({time.time()-t0:.0f}s)", end="")
    print()
    cols = ['standard', 'mixup', 'dropout', 'input_noise', 'born_again', 'KD']
    print(f"\n{'Model':<10}" + "".join(f"{c:<13}" for c in cols))
    print("-" * (10 + 13 * len(cols)))
    summ = {}
    for name in MLP_CONFIGS:
        summ[name] = {c: float(np.mean(res[name][c])) for c in cols}
        print(f"{name:<10}" + "".join(f"{summ[name][c]:<13.3f}" for c in cols))
    DUMP['R8'] = summ
    return summ


# ============================================================
# R9: ADDITIONAL REAL DATASETS (OpenML, network-guarded)   (R1.14)
# ============================================================
def exp_R9():
    print("\n" + "=" * 80)
    print(f"R9  ADDITIONAL REAL DATASETS (OpenML; needs network) (seeds={min(N_SEEDS,10)})")
    print("=" * 80)
    from sklearn.datasets import fetch_openml
    catalog = {  # name -> openml data_id (regression)
        'concrete': 4353, 'energy_efficiency': 44960, 'abalone': 44956,
    }
    n = min(N_SEEDS, 10)
    out = {}
    for dsname, did in catalog.items():
        print(f"\n--- {dsname} (openml {did}) ---")
        try:
            ds = fetch_openml(data_id=did, as_frame=True)
            X = ds.data.select_dtypes('number').to_numpy(float)
            y = np.asarray(ds.target, float)
            X = X[~np.isnan(y)]; y = y[~np.isnan(y)]
        except Exception as e:
            print(f"  [skip] could not fetch ({e})")
            out[dsname] = {'skipped': str(e)}
            continue
        all_data = []
        for s in range(n):
            seed = 42 + s
            sc = StandardScaler(); Xs = sc.fit_transform(X)
            ys = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
            data = _split_real(Xs, ys, seed)
            pe, *_ = train_teacher_ensemble(data, TEACHER_CONFIGS['base'], seed)
            t_tr, t_te = pe(data['X_tr']), pe(data['X_te'])
            r, tim = distill_students(data, seed, t_tr, t_te)
            all_data.append((r, make_meta(data, t_tr, t_te, student_timing=tim)))
            print(f"  seed {s+1}/{n} done")
        agg = aggregate_seeds(all_data)
        out[dsname] = {'summary': summarise(agg), 'aggregate': agg,
                       'teacher_mse': float(agg['_teacher']['mse_mean'])}
        sm = out[dsname]['summary']
        print(f"  teacher MSE={out[dsname]['teacher_mse']:.3f}  helped={sm['helped']}/{sm['n_models']}  "
              f"avg={sm['avg_improv_pct']:+.2f}%  MLP={sm['mlp_improv_pct']:+.2f}%")
    DUMP['R9'] = {k: (v.get('summary') | {'teacher_mse': v['teacher_mse']}
                      if 'summary' in v else v) for k, v in out.items()}
    DUMP['R9_full'] = {k: v.get('aggregate') for k, v in out.items() if 'aggregate' in v}
    return out


def _split_real(X, y, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.15, random_state=seed)
    return dict(X_tr=X_tr, X_va=X_va, X_te=X_te, y_tr=y_tr, y_va=y_va, y_te=y_te,
                yt_tr=None, yt_te=None, n_features=X.shape[1])


# ============================================================
# MAIN
# ============================================================
EXPERIMENTS = {
    'R1': exp_R1, 'R2': exp_R2, 'R3': exp_R3, 'R4': exp_R4, 'R5': exp_R5,
    'R6': exp_R6, 'R7': exp_R7, 'R8': exp_R8, 'R9': exp_R9,
}

if __name__ == '__main__':
    t_start = time.time()
    print("=" * 80)
    print("REGRESSION KD — REVIEWER-RESPONSE EXPERIMENTS")
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

    with open('results_regression_revision_full.json', 'w') as f:
        json.dump(_native(DUMP), f, indent=2)
    print("\nWrote results_regression_revision_full.json")

    print(f"\n{'=' * 80}\nDONE in {(time.time()-t_start)/60:.1f} min\n{'=' * 80}")