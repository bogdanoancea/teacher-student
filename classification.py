"""
Enhanced Knowledge Distillation for Classification
====================================================
Experiments:
  A. Multi-seed main experiment (CIs + paired tests, val-based alpha/T selection)
  B. Label-noise sweep (eta grid)
  C. Teacher strength ablation (weak / base / strong)
  D. Single vs ensemble teacher ablation
  E. Real datasets (Breast Cancer, Wine, Digits)
  F. Label smoothing baseline comparison for MLPs
  G. Temperature sweep (multi-seed)
  H. Calibration: ECE for teacher + key students

Usage:
  python classification.py --quick   # 3 seeds, reduced grids (~20 min)
  python classification.py           # 10 seeds (~3-4 hours)
  python classification.py --full    # 20 seeds (~8+ hours)
"""

import os, sys, warnings, time, json, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from xgboost import XGBClassifier
from scipy import stats
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

# ============================================================
# CONFIGURATION
# ============================================================
if '--full' in sys.argv:
    MODE = 'FULL'; N_SEEDS = 20
elif '--quick' in sys.argv:
    MODE = 'QUICK'; N_SEEDS = 3
else:
    MODE = 'DEFAULT'; N_SEEDS = 10

N_SAMPLES = 8000
N_CLASSES = 5
N_FEATURES = 6
BASE_ETA = 0.10
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]
TEMPERATURES = [1.0, 5.0]  # used during alpha sweep for MLPs

ETA_GRID = {
    'QUICK':   [0.0, 0.10, 0.40],
    'DEFAULT': [0.0, 0.05, 0.10, 0.20, 0.40],
    'FULL':    [0.0, 0.05, 0.10, 0.20, 0.40, 0.60],
}[MODE]

TEACHER_CONFIGS = {
    'weak':   {'nn_layers': [64, 32],    'nn_epochs': 100, 'xgb_n': 50,  'xgb_d': 3, 'gb_n': 50,  'gb_d': 3},
    'base':   {'nn_layers': [512, 512, 256, 256, 128, 64], 'nn_epochs': 300, 'xgb_n': 500, 'xgb_d': 8, 'gb_n': 300, 'gb_d': 6},
    'strong': {'nn_layers': [1024, 512, 512, 256, 256, 128, 64], 'nn_epochs': 300, 'xgb_n': 1000, 'xgb_d': 10, 'gb_n': 800, 'gb_d': 8},
}

MLP_EPOCHS = 80
MLP_BATCH = 128
TEMP_GRID = [1.0, 3.0, 5.0, 10.0, 20.0]

# Label smoothing epsilon values for baseline
LS_EPSILONS = [0.01, 0.05, 0.1, 0.2]


# ============================================================
# DATA GENERATION
# ============================================================
def generate_synthetic(seed, eta=BASE_ETA):
    rng = np.random.RandomState(seed)
    X = rng.randn(N_SAMPLES, N_FEATURES)

    logits = np.zeros((N_SAMPLES, N_CLASSES))
    logits[:, 0] = 2.0*X[:,0]**2 - 1.5*X[:,1] + 0.5*np.sin(np.pi*X[:,2])
    logits[:, 1] = -X[:,0] + 2.0*np.sin(np.pi*X[:,1]) + X[:,2]*X[:,3]
    logits[:, 2] = 1.5*X[:,1]*X[:,2] - X[:,0]**2 + 0.8*np.exp(-X[:,3]**2)
    logits[:, 3] = -0.5*X[:,2]**2 + 1.5*X[:,3] + np.sin(np.pi*X[:,4])*X[:,0]
    logits[:, 4] = X[:,4]**2 - X[:,5]*X[:,0] + 0.5*np.cos(np.pi*X[:,1])

    exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
    true_probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    y_true = true_probs.argmax(axis=1)

    y = y_true.copy()
    if eta > 0:
        n_flip = int(eta * N_SAMPLES)
        flip_idx = rng.choice(N_SAMPLES, size=n_flip, replace=False)
        for idx in flip_idx:
            y[idx] = rng.choice([c for c in range(N_CLASSES) if c != y_true[idx]])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    n_classes = N_CLASSES

    X_tr, X_te, y_tr, y_te, yt_tr, yt_te = train_test_split(
        X, y, y_true, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va, yt_tr, yt_va = train_test_split(
        X_tr, y_tr, yt_tr, test_size=0.15, random_state=seed, stratify=y_tr)

    return dict(X_tr=X_tr, X_va=X_va, X_te=X_te,
                y_tr=y_tr, y_va=y_va, y_te=y_te,
                yt_tr=yt_tr, yt_te=yt_te,
                n_features=X.shape[1], n_classes=n_classes)


def load_real_dataset(name, seed):
    if name == 'breast_cancer':
        data = load_breast_cancer()
    elif name == 'wine':
        data = load_wine()
    elif name == 'digits':
        data = load_digits()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    n_classes = len(np.unique(y))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=seed, stratify=y_tr)

    return dict(X_tr=X_tr, X_va=X_va, X_te=X_te,
                y_tr=y_tr, y_va=y_va, y_te=y_te,
                yt_tr=None, yt_te=None,
                n_features=X.shape[1], n_classes=n_classes)


# ============================================================
# TEACHER
# ============================================================
def create_teacher_nn(layer_sizes, input_dim, n_classes):
    ml = []
    for i, u in enumerate(layer_sizes):
        ml.append(layers.Dense(u, activation='relu',
                               **(dict(input_shape=(input_dim,)) if i == 0 else {})))
        if u >= 256:
            ml.append(layers.BatchNormalization())
            ml.append(layers.Dropout(0.15))
    ml.append(layers.Dense(n_classes, activation='softmax'))
    return keras.Sequential(ml)


def train_teacher_ensemble(data, config, seed):
    tf.random.set_seed(seed)
    X_tr, y_tr, X_va, y_va = data['X_tr'], data['y_tr'], data['X_va'], data['y_va']
    nf, nc = data['n_features'], data['n_classes']

    nn = create_teacher_nn(config['nn_layers'], nf, nc)
    nn.compile(optimizer=keras.optimizers.Adam(0.001),
               loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    nn.fit(X_tr, y_tr, epochs=config['nn_epochs'], batch_size=32,
           validation_data=(X_va, y_va),
           callbacks=[keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True),
                      keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8)],
           verbose=0)

    xgb = XGBClassifier(n_estimators=config['xgb_n'], max_depth=config['xgb_d'],
                         learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         num_class=nc, objective='multi:softprob',
                         random_state=seed, verbosity=0, use_label_encoder=False)
    xgb.fit(X_tr, y_tr)

    gb = GradientBoostingClassifier(n_estimators=config['gb_n'], max_depth=config['gb_d'],
                                     learning_rate=0.05, subsample=0.8, random_state=seed)
    gb.fit(X_tr, y_tr)

    def pred_ens(X):  return (nn.predict(X, verbose=0) + xgb.predict_proba(X) + gb.predict_proba(X)) / 3
    def pred_nn(X):   return nn.predict(X, verbose=0)
    def pred_xgb(X):  return xgb.predict_proba(X)
    def pred_gb(X):   return gb.predict_proba(X)

    return pred_ens, pred_nn, pred_xgb, pred_gb


# ============================================================
# STUDENTS
# ============================================================
def get_sklearn_students(seed):
    return {
        "RF(50,d5)":  RandomForestClassifier(n_estimators=50, max_depth=5, random_state=seed, n_jobs=-1),
        "RF(20,d3)":  RandomForestClassifier(n_estimators=20, max_depth=3, random_state=seed, n_jobs=-1),
        "XGB(sm)":    XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                     random_state=seed, verbosity=0, use_label_encoder=False),
        "GB(sm)":     GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=seed),
        "DT(d5)":     DecisionTreeClassifier(max_depth=5, random_state=seed),
        "DT(d3)":     DecisionTreeClassifier(max_depth=3, random_state=seed),
        "KNN(k5)":    KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "KNN(k10)":   KNeighborsClassifier(n_neighbors=10, weights='distance'),
        "SVM(rbf)":   SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=seed),
        "Logistic":   LogisticRegression(max_iter=1000, random_state=seed),
    }


MLP_CONFIGS = {
    "MLP(8)":    (8, 1),
    "MLP(16)":   (16, 1),
    "MLP(32)":   (32, 1),
    "MLP(16x2)": (16, 2),
    "MLP(32x2)": (32, 2),
}


def create_small_mlp(units, num_layers, input_dim, n_classes):
    ll = [layers.Dense(units, activation='relu', input_shape=(input_dim,))]
    for _ in range(num_layers - 1):
        ll.append(layers.Dense(units, activation='relu'))
    ll.append(layers.Dense(n_classes))  # logits
    return keras.Sequential(ll)


def train_mlp_distilled(X_tr, y_tr, teacher_soft, X_va, y_va, X_te, y_te,
                        units, num_layers, input_dim, n_classes, alpha, temperature, seed):
    """Train single MLP with soft-label distillation. Returns (val_acc, test_acc, test_probs)."""
    tf.random.set_seed(seed)
    model = create_small_mlp(units, num_layers, input_dim, n_classes)
    optimizer = keras.optimizers.Adam(0.001)

    dataset = tf.data.Dataset.from_tensor_slices((
        X_tr.astype(np.float32), y_tr.astype(np.int64), teacher_soft.astype(np.float32)
    )).shuffle(2048, seed=seed).batch(MLP_BATCH).prefetch(tf.data.AUTOTUNE)

    for _ in range(MLP_EPOCHS):
        for x_b, y_b, s_b in dataset:
            with tf.GradientTape() as tape:
                logits = model(x_b, training=True)
                hard_loss = tf.reduce_mean(
                    keras.losses.sparse_categorical_crossentropy(y_b, tf.nn.softmax(logits)))
                soft_pred = tf.nn.softmax(logits / temperature)
                soft_teacher = tf.nn.softmax(tf.math.log(s_b + 1e-10) / temperature)
                kl_loss = keras.losses.KLDivergence()(soft_teacher, soft_pred)
                loss = alpha * hard_loss + (1.0 - alpha) * (temperature**2) * kl_loss
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    val_preds = model.predict(X_va, verbose=0)
    test_preds = model.predict(X_te, verbose=0)
    val_acc = accuracy_score(y_va, val_preds.argmax(axis=1))
    test_acc = accuracy_score(y_te, test_preds.argmax(axis=1))
    # softmax for calibration
    test_probs = tf.nn.softmax(test_preds).numpy()
    return val_acc, test_acc, test_probs


# ============================================================
# CALIBRATION: Expected Calibration Error
# ============================================================
def compute_ece(y_true, probs, n_bins=15):
    """Compute Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            ece += in_bin.sum() * abs(avg_acc - avg_conf)
    return ece / len(y_true)


# ============================================================
# CORE: run one seed
# ============================================================
def run_one_seed(data, seed, teacher_cfg_name='base', teacher_type='ensemble'):
    X_tr, X_va, X_te = data['X_tr'], data['X_va'], data['X_te']
    y_tr, y_va, y_te = data['y_tr'], data['y_va'], data['y_te']
    yt_tr = data.get('yt_tr')
    nf, nc = data['n_features'], data['n_classes']

    cfg = TEACHER_CONFIGS[teacher_cfg_name]
    t0 = time.time()
    pred_ens, pred_nn, pred_xgb, pred_gb = train_teacher_ensemble(data, cfg, seed)
    teacher_time = time.time() - t0

    teacher_fn = {'ensemble': pred_ens, 'nn_only': pred_nn,
                  'xgb_only': pred_xgb, 'gb_only': pred_gb}.get(teacher_type, pred_ens)

    t_soft_tr = teacher_fn(X_tr)
    t_soft_te = teacher_fn(X_te)
    t_hard_tr = t_soft_tr.argmax(axis=1)
    t_preds_te = t_soft_te.argmax(axis=1)

    teacher_acc = accuracy_score(y_te, t_preds_te)
    teacher_ll = log_loss(y_te, t_soft_te)
    teacher_ece = compute_ece(y_te, t_soft_te)
    teacher_entropy = -np.sum(t_soft_tr * np.log(t_soft_tr + 1e-10), axis=1).mean()

    correction_rate = None
    if yt_tr is not None:
        noisy_correct = accuracy_score(yt_tr, y_tr)
        teacher_correct = accuracy_score(yt_tr, t_hard_tr)
        correction_rate = teacher_correct - noisy_correct

    results = {}
    timing = {}

    # ---- sklearn students (hard-label distillation) ----
    students = get_sklearn_students(seed)
    for name, model in students.items():
        results[name] = {}
        t0s = time.time()
        for alpha in ALPHAS:
            m = clone(model)
            if alpha == 1.0:
                m.fit(X_tr, y_tr)
            elif alpha == 0.0:
                m.fit(X_tr, t_hard_tr)
            else:
                blend_rng = np.random.RandomState(seed)
                use_teacher = blend_rng.random(len(y_tr)) > alpha
                blended = np.where(use_teacher, t_hard_tr, y_tr)
                m.fit(X_tr, blended)
            va = accuracy_score(y_va, m.predict(X_va))
            te = accuracy_score(y_te, m.predict(X_te))
            results[name][alpha] = {'val': va, 'test': te}
        timing[name] = time.time() - t0s

    # ---- MLP students (soft-label distillation) ----
    for name, (units, nl) in MLP_CONFIGS.items():
        results[name] = {}
        t0s = time.time()

        # alpha=1.0: standard training
        tf.random.set_seed(seed)
        model = create_small_mlp(units, nl, nf, nc)
        model.compile(optimizer='adam',
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        model.fit(X_tr.astype(np.float32), y_tr.astype(np.int64),
                  epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
        va = accuracy_score(y_va, model.predict(X_va, verbose=0).argmax(axis=1))
        te = accuracy_score(y_te, model.predict(X_te, verbose=0).argmax(axis=1))
        results[name][1.0] = {'val': va, 'test': te}

        # Distilled: search temperature on val
        for alpha in [0.0, 0.3, 0.5, 0.7]:
            best_val, best_test = 0, 0
            for temp in TEMPERATURES:
                va2, te2, _ = train_mlp_distilled(
                    X_tr, y_tr, t_soft_tr, X_va, y_va, X_te, y_te,
                    units, nl, nf, nc, alpha, temp, seed)
                if va2 > best_val:
                    best_val, best_test = va2, te2
            results[name][alpha] = {'val': best_val, 'test': best_test}

        timing[name] = time.time() - t0s

    meta = {
        'teacher_acc': teacher_acc, 'teacher_ll': teacher_ll,
        'teacher_ece': teacher_ece, 'teacher_entropy': teacher_entropy,
        'correction_rate': correction_rate,
        'teacher_time': teacher_time, 'student_timing': timing,
    }
    return results, meta


# ============================================================
# AGGREGATION
# ============================================================
def aggregate_seeds(all_data):
    model_names = list(all_data[0][0].keys())
    out = {}

    for name in model_names:
        base_tests, best_tests, best_alphas = [], [], []
        best_vals = []
        per_alpha_tests = {a: [] for a in ALPHAS}
        per_alpha_vals = {a: [] for a in ALPHAS}

        for results, meta in all_data:
            res = results[name]
            base_tests.append(res[1.0]['test'])
            best_a = max(ALPHAS, key=lambda a: res[a]['val'])
            best_tests.append(res[best_a]['test'])
            best_vals.append(res[best_a]['val'])
            best_alphas.append(best_a)
            for a in ALPHAS:
                per_alpha_tests[a].append(res[a]['test'])
                per_alpha_vals[a].append(res[a]['val'])

        base_arr = np.array(base_tests)
        best_arr = np.array(best_tests)
        delta_pp = (best_arr - base_arr) * 100  # percentage points (positive = helped)

        n = len(delta_pp)
        if n >= 10:
            try: stat, pval = stats.wilcoxon(delta_pp, alternative='greater')
            except ValueError: stat, pval = 0, 1.0
            tname = 'Wilcoxon'
        else:
            stat, pval = stats.ttest_rel(best_arr, base_arr)
            tname = 'paired-t'

        if n > 1:
            ci = stats.t.interval(0.95, n-1, loc=delta_pp.mean(), scale=stats.sem(delta_pp))
        else:
            ci = (delta_pp.mean(), delta_pp.mean())

        out[name] = {
            'base_mean': base_arr.mean(), 'base_std': base_arr.std(),
            'best_mean': best_arr.mean(), 'best_std': best_arr.std(),
            'delta_pp': delta_pp.mean(), 'delta_pp_std': delta_pp.std(),
            'delta_ci': ci,
            'imp_pct': ((best_arr - base_arr) / base_arr).mean() * 100,
            'wins': np.mean(delta_pp > 0) * 100,
            'pval': pval, 'test_name': tname,
            'alpha_mode': float(stats.mode(best_alphas, keepdims=False).mode),
            'alpha_mean': np.mean(best_alphas),
            'per_alpha': {a: (np.mean(per_alpha_tests[a]), np.std(per_alpha_tests[a])) for a in ALPHAS},
            # Raw per-seed values (one entry per seed) for reproducibility / re-formatting.
            'per_seed': {
                'seeds': [42 + i for i in range(n)],
                'base_test_acc': base_tests,
                'best_test_acc': best_tests,
                'best_val_acc': best_vals,
                'best_alpha': best_alphas,
                'delta_pp': delta_pp.tolist(),
                'per_alpha_test_acc': {a: per_alpha_tests[a] for a in ALPHAS},
                'per_alpha_val_acc': {a: per_alpha_vals[a] for a in ALPHAS},
            },
        }

    # Teacher stats
    t_accs = [m['teacher_acc'] for _, m in all_data]
    t_lls = [m['teacher_ll'] for _, m in all_data]
    t_eces = [m['teacher_ece'] for _, m in all_data]
    t_ents = [m['teacher_entropy'] for _, m in all_data]
    corrs = [m['correction_rate'] for _, m in all_data if m['correction_rate'] is not None]
    ttimes = [m['teacher_time'] for _, m in all_data]
    stimes = {}
    for name in model_names:
        stimes[name] = np.mean([m['student_timing'].get(name, 0) for _, m in all_data])

    out['_teacher'] = {
        'acc_mean': np.mean(t_accs), 'acc_std': np.std(t_accs),
        'll_mean': np.mean(t_lls), 'll_std': np.std(t_lls),
        'ece_mean': np.mean(t_eces), 'ece_std': np.std(t_eces),
        'entropy_mean': np.mean(t_ents),
        'correction': np.mean(corrs) * 100 if corrs else None,
        'train_time': np.mean(ttimes),
        'per_seed': {
            'seeds': [42 + i for i in range(len(t_accs))],
            'teacher_acc': t_accs,
            'teacher_ll': t_lls,
            'teacher_ece': t_eces,
            'teacher_entropy': t_ents,
            # correction_rate is per-seed; None where unavailable (no true labels)
            'correction_rate': [(m['correction_rate'] * 100 if m['correction_rate'] is not None else None)
                                for _, m in all_data],
        },
    }
    out['_timing'] = stimes
    return out


# ============================================================
# PRINTING
# ============================================================
def print_table(agg, title=""):
    if title: print(f"\n{title}")
    print(f"{'Model':<12} {'BaseAcc':<22} {'DistilAcc':<22} {'Delta(pp)':<22} {'Wins%':<7} {'p-val':<10} {'a*':<5}")
    print("-" * 102)
    models = [k for k in agg if not k.startswith('_')]
    for name in sorted(models, key=lambda n: agg[n]['best_mean'], reverse=True):
        a = agg[name]
        sig = '***' if a['pval'] < 0.001 else ('**' if a['pval'] < 0.01 else ('*' if a['pval'] < 0.05 else ''))
        print(f"{name:<12} {a['base_mean']:.4f} +/- {a['base_std']:.4f}  "
              f"{a['best_mean']:.4f} +/- {a['best_std']:.4f}  "
              f"{a['delta_pp']:+5.1f}pp [{a['delta_ci'][0]:+.1f},{a['delta_ci'][1]:+.1f}]  "
              f"{a['wins']:5.0f}%  {a['pval']:.4f}{sig} {a['alpha_mode']:.1f}")


def print_per_alpha_table(agg, title=""):
    if title: print(f"\n{title}")
    print(f"{'Model':<12}", end="")
    for a in ALPHAS:
        print(f" {'a='+str(a):<16}", end="")
    print(f" {'Best a':<6}")
    print("-" * (12 + 17 * len(ALPHAS) + 8))
    models = [k for k in agg if not k.startswith('_')]
    for name in sorted(models, key=lambda n: agg[n]['best_mean'], reverse=True):
        d = agg[name]
        print(f"{name:<12}", end="")
        for a in ALPHAS:
            m, s = d['per_alpha'][a]
            print(f" {m:.4f}+/-{s:.3f} ", end="")
        print(f" {d['alpha_mode']:.1f}")


def print_category_summary(agg, title=""):
    if title: print(f"\n{title}")
    models = {k: v for k, v in agg.items() if not k.startswith('_')}
    cats = {
        'KNN':      [n for n in models if 'KNN' in n],
        'MLP':      [n for n in models if 'MLP' in n],
        'Trees':    [n for n in models if any(x in n for x in ['RF', 'XGB', 'GB', 'DT'])],
        'SVM':      [n for n in models if 'SVM' in n],
        'Logistic': [n for n in models if 'Log' in n],
    }
    print(f"{'Category':<10} {'Avg a*':<8} {'Avg Delta(pp)':<16} {'Avg Wins%':<10}")
    print("-" * 46)
    for cat, names in cats.items():
        if names:
            aa = np.mean([models[n]['alpha_mode'] for n in names])
            ad = np.mean([models[n]['delta_pp'] for n in names])
            aw = np.mean([models[n]['wins'] for n in names])
            print(f"{cat:<10} {aa:<8.2f} {ad:<16.1f} {aw:<10.0f}")


def print_timing(agg, title=""):
    if title: print(f"\n{title}")
    t = agg['_teacher']
    print(f"  Teacher training: {t['train_time']:.1f}s")
    print(f"  Students (mean over alpha sweep):")
    for name, st in sorted(agg['_timing'].items(), key=lambda x: x[1]):
        print(f"    {name:<12} {st:.2f}s")


# ============================================================
# EXPERIMENT A
# ============================================================
def experiment_A():
    print("\n" + "=" * 80)
    print(f"EXPERIMENT A: MAIN MULTI-SEED (seeds={N_SEEDS}, eta={BASE_ETA})")
    print("Test set NEVER used for alpha/T selection.")
    print("=" * 80)

    all_data = []
    for s in range(N_SEEDS):
        seed = 42 + s
        t0 = time.time()
        print(f"  Seed {s+1}/{N_SEEDS} (seed={seed})...", end=" ", flush=True)
        data = generate_synthetic(seed, BASE_ETA)
        results, meta = run_one_seed(data, seed, 'base', 'ensemble')
        all_data.append((results, meta))
        print(f"done ({time.time()-t0:.0f}s)")

    agg = aggregate_seeds(all_data)

    t = agg['_teacher']
    print(f"\n--- Teacher Performance ({N_SEEDS} seeds) ---")
    print(f"  Accuracy:     {t['acc_mean']:.4f} +/- {t['acc_std']:.4f}")
    print(f"  Log loss:     {t['ll_mean']:.4f} +/- {t['ll_std']:.4f}")
    print(f"  ECE:          {t['ece_mean']:.4f} +/- {t['ece_std']:.4f}")
    print(f"  Entropy:      {t['entropy_mean']:.4f} nats")
    if t['correction'] is not None:
        print(f"  Correction:   {t['correction']:.1f}%")

    print_per_alpha_table(agg, "--- Per-Alpha Accuracy (mean +/- std) ---")
    print_table(agg, "--- Distillation Benefit (val-selected a vs baseline a=1.0) ---")
    print_category_summary(agg, "--- Category Summary ---")
    print_timing(agg, "--- Computational Cost ---")

    models = {k: v for k, v in agg.items() if not k.startswith('_')}
    helped = sum(1 for v in models.values() if v['delta_pp'] > 0)
    avg_d = np.mean([v['delta_pp'] for v in models.values()])
    print(f"\nOverall: {helped}/{len(models)} improved, avg delta: {avg_d:+.1f}pp")

    return agg, all_data


# ============================================================
# EXPERIMENT B: Noise sweep
# ============================================================
def experiment_B():
    print("\n" + "=" * 80)
    print(f"EXPERIMENT B: LABEL NOISE SWEEP (etas={[f'{e*100:.0f}%' for e in ETA_GRID]}, seeds={N_SEEDS})")
    print("=" * 80)

    sweep = {}
    for eta in ETA_GRID:
        print(f"\n--- eta = {eta*100:.0f}% ---")
        all_data = []
        for s in range(N_SEEDS):
            seed = 42 + s
            print(f"  Seed {s+1}/{N_SEEDS}...", end=" ", flush=True)
            t0 = time.time()
            data = generate_synthetic(seed, eta)
            results, meta = run_one_seed(data, seed, 'base', 'ensemble')
            all_data.append((results, meta))
            print(f"({time.time()-t0:.0f}s)", end="")
        print()
        sweep[eta] = aggregate_seeds(all_data)

    print("\n" + "-" * 80)
    print("LABEL NOISE SWEEP SUMMARY")
    print(f"{'eta%':<7} {'TeachAcc':<12} {'Correct%':<10} {'Helped':<10} {'AvgDelta(pp)':<14} {'MLP D(pp)':<12}")
    print("-" * 67)
    for eta in ETA_GRID:
        a = sweep[eta]; t = a['_teacher']
        models = {k: v for k, v in a.items() if not k.startswith('_')}
        h = sum(1 for v in models.values() if v['delta_pp'] > 0)
        ad = np.mean([v['delta_pp'] for v in models.values()])
        md = np.mean([v['delta_pp'] for k, v in models.items() if 'MLP' in k])
        cr = f"{t['correction']:.1f}" if t['correction'] is not None else "N/A"
        print(f"{eta*100:<7.0f} {t['acc_mean']:<12.4f} {cr:<10} {h}/{len(models):<7} {ad:<14.1f} {md:<12.1f}")

    # Per-model
    rep = ['KNN(k5)', 'MLP(32x2)', 'MLP(8)', 'GB(sm)', 'Logistic']
    print(f"\n{'Model':<12}", end="")
    for eta in ETA_GRID: print(f" eta={eta*100:>3.0f}%", end="")
    print()
    print("-" * (12 + 10 * len(ETA_GRID)))
    for mn in rep:
        print(f"{mn:<12}", end="")
        for eta in ETA_GRID:
            if mn in sweep[eta]:
                print(f" {sweep[eta][mn]['delta_pp']:>+5.1f}pp", end="")
            else:
                print(f"    N/A  ", end="")
        print()

    return sweep


# ============================================================
# EXPERIMENT C: Teacher strength ablation
# ============================================================
def experiment_C():
    print("\n" + "=" * 80)
    print(f"EXPERIMENT C: TEACHER STRENGTH ABLATION (seeds={N_SEEDS})")
    print("=" * 80)

    ablation = {}
    for tname in ['weak', 'base', 'strong']:
        print(f"\n--- Teacher: {tname} ---")
        all_data = []
        for s in range(N_SEEDS):
            seed = 42 + s
            print(f"  Seed {s+1}/{N_SEEDS}...", end=" ", flush=True)
            t0 = time.time()
            data = generate_synthetic(seed, BASE_ETA)
            results, meta = run_one_seed(data, seed, tname, 'ensemble')
            all_data.append((results, meta))
            print(f"({time.time()-t0:.0f}s)", end="")
        print()
        ablation[tname] = aggregate_seeds(all_data)

    print("\n" + "-" * 80)
    print("TEACHER ABLATION SUMMARY")
    print(f"{'Teacher':<8} {'Acc':<12} {'ECE':<10} {'Correct%':<10} {'Entropy':<10} {'Helped':<10} {'AvgD(pp)':<10}")
    print("-" * 72)
    for tname in ['weak', 'base', 'strong']:
        a = ablation[tname]; t = a['_teacher']
        models = {k: v for k, v in a.items() if not k.startswith('_')}
        h = sum(1 for v in models.values() if v['delta_pp'] > 0)
        ad = np.mean([v['delta_pp'] for v in models.values()])
        cr = f"{t['correction']:.1f}" if t['correction'] is not None else "N/A"
        print(f"{tname:<8} {t['acc_mean']:<12.4f} {t['ece_mean']:<10.4f} {cr:<10} "
              f"{t['entropy_mean']:<10.4f} {h}/{len(models):<7} {ad:<10.1f}")

    # Per-model
    rep = ['KNN(k5)', 'MLP(32x2)', 'MLP(8)', 'GB(sm)', 'SVM(rbf)']
    print(f"\n{'Model':<12} {'Weak D(pp)':<14} {'Base D(pp)':<14} {'Strong D(pp)':<14}")
    print("-" * 56)
    for mn in rep:
        print(f"{mn:<12}", end="")
        for tname in ['weak', 'base', 'strong']:
            if mn in ablation[tname]:
                print(f" {ablation[tname][mn]['delta_pp']:>+6.1f}pp     ", end="")
            else:
                print(f" {'N/A':>6}       ", end="")
        print()

    # Alpha shift
    print(f"\n{'Model':<12} {'Weak a*':<10} {'Base a*':<10} {'Strong a*':<10}")
    print("-" * 42)
    for mn in rep:
        print(f"{mn:<12}", end="")
        for tname in ['weak', 'base', 'strong']:
            if mn in ablation[tname]:
                print(f" {ablation[tname][mn]['alpha_mode']:>5.1f}    ", end="")
            else:
                print(f" {'N/A':>5}     ", end="")
        print()

    return ablation


# ============================================================
# EXPERIMENT D: Single vs ensemble teacher
# ============================================================
def experiment_D():
    print("\n" + "=" * 80)
    print(f"EXPERIMENT D: SINGLE VS ENSEMBLE TEACHER (seeds={N_SEEDS})")
    print("=" * 80)

    types = ['nn_only', 'xgb_only', 'gb_only', 'ensemble']
    ablation = {}
    for ttype in types:
        print(f"\n--- Teacher type: {ttype} ---")
        all_data = []
        for s in range(N_SEEDS):
            seed = 42 + s
            print(f"  Seed {s+1}/{N_SEEDS}...", end=" ", flush=True)
            t0 = time.time()
            data = generate_synthetic(seed, BASE_ETA)
            results, meta = run_one_seed(data, seed, 'base', ttype)
            all_data.append((results, meta))
            print(f"({time.time()-t0:.0f}s)", end="")
        print()
        ablation[ttype] = aggregate_seeds(all_data)

    print("\n" + "-" * 80)
    print("SINGLE VS ENSEMBLE TEACHER")
    print(f"{'Type':<12} {'TeachAcc':<12} {'Helped':<10} {'AvgD(pp)':<12} {'MLP D(pp)':<12}")
    print("-" * 58)
    for ttype in types:
        a = ablation[ttype]; t = a['_teacher']
        models = {k: v for k, v in a.items() if not k.startswith('_')}
        h = sum(1 for v in models.values() if v['delta_pp'] > 0)
        ad = np.mean([v['delta_pp'] for v in models.values()])
        md = np.mean([v['delta_pp'] for k, v in models.items() if 'MLP' in k])
        print(f"{ttype:<12} {t['acc_mean']:<12.4f} {h}/{len(models):<7} {ad:<12.1f} {md:<12.1f}")

    return ablation


# ============================================================
# EXPERIMENT E: Real datasets
# ============================================================
def experiment_E():
    print("\n" + "=" * 80)
    print(f"EXPERIMENT E: REAL DATASETS (seeds={N_SEEDS})")
    print("=" * 80)

    datasets = ['breast_cancer', 'wine', 'digits']
    real_results = {}

    for dsname in datasets:
        print(f"\n--- Dataset: {dsname} ---")
        all_data = []
        for s in range(N_SEEDS):
            seed = 42 + s
            print(f"  Seed {s+1}/{N_SEEDS}...", end=" ", flush=True)
            t0 = time.time()
            data = load_real_dataset(dsname, seed)
            results, meta = run_one_seed(data, seed, 'base', 'ensemble')
            all_data.append((results, meta))
            print(f"({time.time()-t0:.0f}s)", end="")
        print()
        agg = aggregate_seeds(all_data)
        real_results[dsname] = agg
        print_table(agg, f"--- {dsname}: Results ---")

    print("\n" + "-" * 80)
    print("REAL DATASETS CROSS-SUMMARY")
    print(f"{'Dataset':<16} {'TeachAcc':<12} {'Helped':<10} {'AvgD(pp)':<12} {'MLP D(pp)':<12} {'KNN D(pp)':<12}")
    print("-" * 74)
    for dsname in datasets:
        a = real_results[dsname]; t = a['_teacher']
        models = {k: v for k, v in a.items() if not k.startswith('_')}
        h = sum(1 for v in models.values() if v['delta_pp'] > 0)
        ad = np.mean([v['delta_pp'] for v in models.values()])
        md = np.mean([v['delta_pp'] for k, v in models.items() if 'MLP' in k])
        kd = np.mean([v['delta_pp'] for k, v in models.items() if 'KNN' in k])
        print(f"{dsname:<16} {t['acc_mean']:<12.4f} {h}/{len(models):<7} {ad:<12.1f} {md:<12.1f} {kd:<12.1f}")

    return real_results


# ============================================================
# EXPERIMENT F: Label smoothing baseline
# ============================================================
def experiment_F(all_data_A):
    print("\n" + "=" * 80)
    print(f"EXPERIMENT F: LABEL SMOOTHING BASELINE vs KD (seeds={N_SEEDS})")
    print("=" * 80)

    baseline_results = {}
    for s in range(N_SEEDS):
        seed = 42 + s
        print(f"  Seed {s+1}/{N_SEEDS}...", end=" ", flush=True)
        t0 = time.time()
        data = generate_synthetic(seed, BASE_ETA)
        X_tr, y_tr = data['X_tr'], data['y_tr']
        X_te, y_te = data['X_te'], data['y_te']
        nf, nc = data['n_features'], data['n_classes']

        bl = {}
        for name, (units, nl) in MLP_CONFIGS.items():
            bl[name] = {}

            # Standard
            tf.random.set_seed(seed)
            m = create_small_mlp(units, nl, nf, nc)
            m.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            m.fit(X_tr.astype(np.float32), y_tr.astype(np.int64),
                  epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            bl[name]['standard'] = accuracy_score(y_te, m.predict(X_te, verbose=0).argmax(axis=1))

            # Label smoothing with different epsilons
            for eps in LS_EPSILONS:
                tf.random.set_seed(seed)
                m = create_small_mlp(units, nl, nf, nc)
                m.compile(optimizer='adam',
                          loss=keras.losses.CategoricalCrossentropy(from_logits=True,
                                                                     label_smoothing=eps))
                y_onehot = keras.utils.to_categorical(y_tr, nc)
                m.fit(X_tr.astype(np.float32), y_onehot,
                      epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
                bl[name][f'LS({eps})'] = accuracy_score(y_te, m.predict(X_te, verbose=0).argmax(axis=1))

            # Early stopping (50% epochs)
            tf.random.set_seed(seed)
            m = create_small_mlp(units, nl, nf, nc)
            m.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            m.fit(X_tr.astype(np.float32), y_tr.astype(np.int64),
                  epochs=MLP_EPOCHS // 2, batch_size=MLP_BATCH, verbose=0)
            bl[name]['EarlyStop'] = accuracy_score(y_te, m.predict(X_te, verbose=0).argmax(axis=1))

        baseline_results[seed] = bl
        print(f"({time.time()-t0:.0f}s)")

    print(f"\n{'MLP':<12} {'Standard':<14} {'BestLS':<14} {'EarlyStop':<14} {'BestKD':<14}")
    print("-" * 68)
    for name in MLP_CONFIGS:
        std = [baseline_results[42+s][name]['standard'] for s in range(N_SEEDS)]
        es = [baseline_results[42+s][name]['EarlyStop'] for s in range(N_SEEDS)]
        bls = []
        for s in range(N_SEEDS):
            ls_vals = {k: v for k, v in baseline_results[42+s][name].items() if k.startswith('LS(')}
            bls.append(max(ls_vals.values()))
        kd = []
        for results, meta in all_data_A:
            res = results[name]
            best_a = max(ALPHAS, key=lambda a: res[a]['val'])
            kd.append(res[best_a]['test'])

        print(f"{name:<12} "
              f"{np.mean(std):.4f}+/-{np.std(std):.3f} "
              f"{np.mean(bls):.4f}+/-{np.std(bls):.3f} "
              f"{np.mean(es):.4f}+/-{np.std(es):.3f} "
              f"{np.mean(kd):.4f}+/-{np.std(kd):.3f}")

    return baseline_results


# ============================================================
# EXPERIMENT G: Temperature sweep (multi-seed)
# ============================================================
def experiment_G():
    print("\n" + "=" * 80)
    print(f"EXPERIMENT G: TEMPERATURE SWEEP (MLP 32x2, a=0.3, seeds={N_SEEDS})")
    print("=" * 80)

    temp_results = {T: [] for T in TEMP_GRID}
    for s in range(N_SEEDS):
        seed = 42 + s
        print(f"  Seed {s+1}/{N_SEEDS}...", end=" ", flush=True)
        t0 = time.time()
        data = generate_synthetic(seed, BASE_ETA)
        cfg = TEACHER_CONFIGS['base']
        pred_ens, _, _, _ = train_teacher_ensemble(data, cfg, seed)
        t_soft = pred_ens(data['X_tr'])
        for T in TEMP_GRID:
            _, te, _ = train_mlp_distilled(
                data['X_tr'], data['y_tr'], t_soft,
                data['X_va'], data['y_va'], data['X_te'], data['y_te'],
                32, 2, data['n_features'], data['n_classes'], 0.3, T, seed)
            temp_results[T].append(te)
        print(f"({time.time()-t0:.0f}s)")

    print(f"\n{'Temp':<8} {'Mean Acc':<14} {'Std':<10} {'95% CI':<24}")
    print("-" * 58)
    for T in TEMP_GRID:
        arr = np.array(temp_results[T])
        if len(arr) > 1:
            ci = stats.t.interval(0.95, len(arr)-1, loc=arr.mean(), scale=stats.sem(arr))
        else:
            ci = (arr.mean(), arr.mean())
        print(f"T={T:<5.0f} {arr.mean():<14.4f} {arr.std():<10.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

    return temp_results


# ============================================================
# EXPERIMENT H: Calibration (ECE) for teacher & key students
# ============================================================
def experiment_H():
    print("\n" + "=" * 80)
    print(f"EXPERIMENT H: CALIBRATION (ECE) ANALYSIS (seeds={N_SEEDS})")
    print("=" * 80)

    eces = {'teacher': [], 'baseline_MLP(32x2)': [], 'distil_MLP(32x2)': [],
            'baseline_MLP(8)': [], 'distil_MLP(8)': []}

    for s in range(N_SEEDS):
        seed = 42 + s
        print(f"  Seed {s+1}/{N_SEEDS}...", end=" ", flush=True)
        t0 = time.time()
        data = generate_synthetic(seed, BASE_ETA)
        X_tr, y_tr = data['X_tr'], data['y_tr']
        X_va, y_va = data['X_va'], data['y_va']
        X_te, y_te = data['X_te'], data['y_te']
        nf, nc = data['n_features'], data['n_classes']

        cfg = TEACHER_CONFIGS['base']
        pred_ens, _, _, _ = train_teacher_ensemble(data, cfg, seed)
        t_soft_tr = pred_ens(X_tr)
        t_soft_te = pred_ens(X_te)
        eces['teacher'].append(compute_ece(y_te, t_soft_te))

        for mname, (units, nl) in [('MLP(32x2)', (32, 2)), ('MLP(8)', (8, 1))]:
            # Baseline
            tf.random.set_seed(seed)
            m = create_small_mlp(units, nl, nf, nc)
            m.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            m.fit(X_tr.astype(np.float32), y_tr.astype(np.int64),
                  epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            base_probs = tf.nn.softmax(m.predict(X_te, verbose=0)).numpy()
            eces[f'baseline_{mname}'].append(compute_ece(y_te, base_probs))

            # Distilled (a=0.3, T=5)
            _, _, dist_probs = train_mlp_distilled(
                X_tr, y_tr, t_soft_tr, X_va, y_va, X_te, y_te,
                units, nl, nf, nc, 0.3, 5.0, seed)
            eces[f'distil_{mname}'].append(compute_ece(y_te, dist_probs))

        print(f"({time.time()-t0:.0f}s)")

    print(f"\n{'Model':<24} {'ECE Mean':<12} {'ECE Std':<10}")
    print("-" * 48)
    for k, vals in eces.items():
        arr = np.array(vals)
        print(f"{k:<24} {arr.mean():<12.4f} {arr.std():<10.4f}")

    return eces


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    t_start = time.time()
    print("=" * 80)
    print("KNOWLEDGE DISTILLATION FOR CLASSIFICATION - ENHANCED EXPERIMENTS")
    print(f"Mode: {MODE} | Seeds: {N_SEEDS} | Etas: {[f'{e*100:.0f}%' for e in ETA_GRID]}")
    print("=" * 80)

    agg_A, all_data_A = experiment_A()
    sweep_B = experiment_B()
    ablation_C = experiment_C()
    single_D = experiment_D()
    real_E = experiment_E()
    baselines_F = experiment_F(all_data_A)
    temp_G = experiment_G()
    ece_H = experiment_H()

    # ------------------------------------------------------------------
    # Persist full-precision results (incl. raw per-seed values) so any
    # table can be reformatted / any statistic recomputed without rerunning.
    # ------------------------------------------------------------------
    def _to_native(obj):
        """Recursively convert numpy types/containers to JSON-native Python."""
        if isinstance(obj, dict):
            return {(k if isinstance(k, str) else _to_native(k)): _to_native(v)
                    for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_native(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    try:
        payload = {
            'config': {
                'mode': MODE, 'n_seeds': N_SEEDS, 'seeds': [42 + i for i in range(N_SEEDS)],
                'n_samples': N_SAMPLES, 'n_classes': N_CLASSES, 'n_features': N_FEATURES,
                'base_eta': BASE_ETA, 'alphas': ALPHAS, 'temperatures': TEMPERATURES,
                'eta_grid': ETA_GRID, 'temp_grid': TEMP_GRID, 'ls_epsilons': LS_EPSILONS,
            },
            'experiment_A': _to_native(agg_A),
            'experiment_B': {str(eta): _to_native(sweep_B[eta]) for eta in sweep_B},
            'experiment_C': {tname: _to_native(ablation_C[tname]) for tname in ablation_C},
            'experiment_D': {ttype: _to_native(single_D[ttype]) for ttype in single_D},
            'experiment_E': {ds: _to_native(real_E[ds]) for ds in real_E},
            'experiment_F': _to_native(baselines_F),     # seed -> model -> {strategy: acc}
            'experiment_G': {str(T): _to_native(temp_G[T]) for T in temp_G},  # T -> [per-seed acc]
            'experiment_H': _to_native(ece_H),           # key -> [per-seed ECE]
        }
        with open('results_classification_full.json', 'w') as f:
            json.dump(payload, f, indent=2)
        print("\nFull-precision results (with per-seed values) written to "
              "results_classification_full.json")
    except Exception as e:
        print(f"\n[warn] could not write JSON dump: {e}")

    # Flat per-seed CSV (experiments A-E): one row per experiment x condition x model x seed.
    def _emit_agg_rows(writer, experiment, condition, agg):
        for model, v in agg.items():
            if model.startswith('_'):
                continue
            ps = v.get('per_seed')
            if not ps:
                continue
            for i, seed in enumerate(ps['seeds']):
                row = {
                    'experiment': experiment,
                    'condition': condition,
                    'model': model,
                    'seed': seed,
                    'base_test_acc': ps['base_test_acc'][i],
                    'best_test_acc': ps['best_test_acc'][i],
                    'best_val_acc': ps['best_val_acc'][i],
                    'best_alpha': ps['best_alpha'][i],
                    'delta_pp': ps['delta_pp'][i],
                }
                for a in ALPHAS:
                    row[f'test_acc_a{a}'] = ps['per_alpha_test_acc'][a][i]
                writer.writerow(row)

    try:
        fieldnames = (['experiment', 'condition', 'model', 'seed',
                       'base_test_acc', 'best_test_acc', 'best_val_acc',
                       'best_alpha', 'delta_pp']
                      + [f'test_acc_a{a}' for a in ALPHAS])
        with open('results_classification_per_seed.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            _emit_agg_rows(w, 'A', 'main', agg_A)
            for eta in sweep_B:
                _emit_agg_rows(w, 'B', f'eta={eta}', sweep_B[eta])
            for tname in ablation_C:
                _emit_agg_rows(w, 'C', f'teacher={tname}', ablation_C[tname])
            for ttype in single_D:
                _emit_agg_rows(w, 'D', f'teacher={ttype}', single_D[ttype])
            for ds in real_E:
                _emit_agg_rows(w, 'E', f'dataset={ds}', real_E[ds])
        print("Per-seed model results written to results_classification_per_seed.csv")
    except Exception as e:
        print(f"[warn] could not write per-seed CSV: {e}")

    # Experiment F: seed -> model -> {strategy: test_acc}. One row per model x seed.
    try:
        any_seed = next(iter(baselines_F))
        any_model = next(iter(baselines_F[any_seed]))
        strategies = list(baselines_F[any_seed][any_model].keys())
        f_fields = ['experiment', 'model', 'seed'] + strategies
        with open('results_classification_baselines_per_seed.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=f_fields)
            w.writeheader()
            for seed, models in baselines_F.items():
                for model, strat_accs in models.items():
                    row = {'experiment': 'F', 'model': model, 'seed': seed}
                    for strat in strategies:
                        row[strat] = strat_accs.get(strat)
                    w.writerow(row)
        print("Per-seed baseline (Exp F) results written to "
              "results_classification_baselines_per_seed.csv")
    except Exception as e:
        print(f"[warn] could not write baselines CSV: {e}")

    # Experiments G (temperature sweep) and H (ECE) are per-seed lists; emit long-format.
    try:
        seeds = [42 + i for i in range(N_SEEDS)]
        with open('results_classification_temp_ece_per_seed.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['experiment', 'metric', 'condition', 'seed', 'value'])
            for T, vals in temp_G.items():
                for i, v in enumerate(vals):
                    w.writerow(['G', 'test_acc', f'T={T}', seeds[i] if i < len(seeds) else 42 + i, v])
            for key, vals in ece_H.items():
                for i, v in enumerate(vals):
                    w.writerow(['H', 'ece', key, seeds[i] if i < len(seeds) else 42 + i, v])
        print("Per-seed temperature/ECE results written to "
              "results_classification_temp_ece_per_seed.csv")
    except Exception as e:
        print(f"[warn] could not write temp/ECE CSV: {e}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"ALL CLASSIFICATION EXPERIMENTS COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'=' * 80}")