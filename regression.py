"""
Enhanced Knowledge Distillation for Regression
================================================
Experiments:
  A. Multi-seed main experiment (CIs + paired tests, val-based alpha selection)
  B. Noise sweep (sigma grid)
  C. Teacher strength ablation (weak / base / strong)
  D. Single vs ensemble teacher ablation
  E. Real datasets (California Housing, Diabetes, Concrete Strength proxy)
  F. Label smoothing / ridge baselines for MLPs
  G. Computational cost tracking

Usage:
  python regression.py --quick   # 3 seeds, reduced grids (~15 min)
  python regression.py           # 10 seeds (~2-3 hours)
  python regression.py --full    # 20 seeds (~5-6 hours)
"""

import os, sys, warnings, time, json, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing, load_diabetes
from xgboost import XGBRegressor
from scipy import stats
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

# ============================================================
# CONFIGURATION
# ============================================================
if '--full' in sys.argv:
    MODE = 'FULL'
    N_SEEDS = 20
elif '--quick' in sys.argv:
    MODE = 'QUICK'
    N_SEEDS = 3
else:
    MODE = 'DEFAULT'
    N_SEEDS = 10

N_SAMPLES = 5000
N_FEATURES = 3
BASE_SIGMA = 0.3
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]

SIGMA_GRID = {
    'QUICK':   [0.1, 0.3, 0.8],
    'DEFAULT': [0.0, 0.1, 0.3, 0.5, 0.8],
    'FULL':    [0.0, 0.1, 0.2, 0.3, 0.5, 0.8],
}[MODE]

TEACHER_CONFIGS = {
    'weak':   {'nn_layers': [64, 32],    'nn_epochs': 100, 'xgb_n': 50,  'xgb_d': 3, 'gb_n': 50,  'gb_d': 3},
    'base':   {'nn_layers': [512, 512, 256, 256, 128, 64], 'nn_epochs': 500, 'xgb_n': 500, 'xgb_d': 8, 'gb_n': 500, 'gb_d': 6},
    'strong': {'nn_layers': [1024, 512, 512, 256, 256, 128, 64], 'nn_epochs': 500, 'xgb_n': 1000, 'xgb_d': 10, 'gb_n': 800, 'gb_d': 8},
}

MLP_EPOCHS = 200
MLP_BATCH = 64

# Label smoothing grid for baseline comparison
LS_SIGMAS = [0.01, 0.05, 0.1, 0.3]  # Gaussian label smoothing stddevs


# ============================================================
# TRUE FUNCTION
# ============================================================
def true_function(X):
    return (2.0 * X[:, 0]**2
            - 3.0 * np.sin(X[:, 1] * np.pi)
            + 1.5 * X[:, 0] * X[:, 2]
            + 0.5 * np.exp(-X[:, 2]**2))


# ============================================================
# DATA GENERATION
# ============================================================
def generate_synthetic(seed, sigma=BASE_SIGMA):
    rng = np.random.RandomState(seed)
    X = rng.randn(N_SAMPLES, N_FEATURES)
    y_true = true_function(X)
    y = y_true + sigma * rng.randn(N_SAMPLES)
    X_tr, X_te, y_tr, y_te, yt_tr, yt_te = train_test_split(
        X, y, y_true, test_size=0.2, random_state=seed)
    X_tr, X_va, y_tr, y_va, yt_tr, yt_va = train_test_split(
        X_tr, y_tr, yt_tr, test_size=0.15, random_state=seed)
    return dict(X_tr=X_tr, X_va=X_va, X_te=X_te,
                y_tr=y_tr, y_va=y_va, y_te=y_te,
                yt_tr=yt_tr, yt_te=yt_te, n_features=N_FEATURES)


def load_real_dataset(name, seed):
    """Load a real regression dataset, standardise, split."""
    if name == 'california':
        data = fetch_california_housing()
    elif name == 'diabetes':
        data = load_diabetes()
    elif name == 'friedman':
        # Friedman #1 synthetic but standard benchmark
        from sklearn.datasets import make_friedman1
        X, y = make_friedman1(n_samples=2000, n_features=10, noise=1.0, random_state=seed)
        class D: pass
        data = D(); data.data = X; data.target = y
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.15, random_state=seed)
    nf = X.shape[1]
    return dict(X_tr=X_tr, X_va=X_va, X_te=X_te,
                y_tr=y_tr, y_va=y_va, y_te=y_te,
                yt_tr=None, yt_te=None, n_features=nf)


# ============================================================
# TEACHER TRAINING
# ============================================================
def create_teacher_nn(layer_sizes, input_dim):
    ml = []
    for i, u in enumerate(layer_sizes):
        ml.append(layers.Dense(u, activation='relu',
                               **(dict(input_shape=(input_dim,)) if i == 0 else {})))
        if u >= 256:
            ml.append(layers.BatchNormalization())
            ml.append(layers.Dropout(0.1))
    ml.append(layers.Dense(1))
    return keras.Sequential(ml)


def train_teacher_ensemble(data, config, seed):
    tf.random.set_seed(seed)
    X_tr, y_tr, X_va, y_va = data['X_tr'], data['y_tr'], data['X_va'], data['y_va']
    nf = data['n_features']

    nn = create_teacher_nn(config['nn_layers'], nf)
    nn.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
    nn.fit(X_tr, y_tr, epochs=config['nn_epochs'], batch_size=32,
           validation_data=(X_va, y_va),
           callbacks=[keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
                      keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)],
           verbose=0)

    xgb = XGBRegressor(n_estimators=config['xgb_n'], max_depth=config['xgb_d'],
                        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        random_state=seed, verbosity=0)
    xgb.fit(X_tr, y_tr)

    gb = GradientBoostingRegressor(n_estimators=config['gb_n'], max_depth=config['gb_d'],
                                    learning_rate=0.05, subsample=0.8, random_state=seed)
    gb.fit(X_tr, y_tr)

    def predict_ensemble(X):
        return (nn.predict(X, verbose=0).flatten() + xgb.predict(X) + gb.predict(X)) / 3

    # also return individual teacher predict fns
    def predict_nn(X): return nn.predict(X, verbose=0).flatten()
    def predict_xgb(X): return xgb.predict(X)
    def predict_gb(X): return gb.predict(X)

    return predict_ensemble, predict_nn, predict_xgb, predict_gb


# ============================================================
# STUDENT DEFINITIONS
# ============================================================
def get_sklearn_students(seed):
    return {
        "RF(50,d5)":  RandomForestRegressor(n_estimators=50, max_depth=5, random_state=seed, n_jobs=-1),
        "RF(20,d3)":  RandomForestRegressor(n_estimators=20, max_depth=3, random_state=seed, n_jobs=-1),
        "XGB(sm)":    XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=seed, verbosity=0),
        "GB(sm)":     GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=seed),
        "DT(d5)":     DecisionTreeRegressor(max_depth=5, random_state=seed),
        "DT(d3)":     DecisionTreeRegressor(max_depth=3, random_state=seed),
        "KNN(k5)":    KNeighborsRegressor(n_neighbors=5, weights='distance'),
        "KNN(k10)":   KNeighborsRegressor(n_neighbors=10, weights='distance'),
        "SVR(rbf)":   SVR(kernel='rbf', C=1.0, gamma='scale'),
    }


MLP_CONFIGS = {
    "MLP(4)":    (4, 1),
    "MLP(8)":    (8, 1),
    "MLP(16)":   (16, 1),
    "MLP(8x2)":  (8, 2),
    "MLP(16x2)": (16, 2),
}


def create_small_mlp(units, num_layers, input_dim):
    ll = [layers.Dense(units, activation='relu', input_shape=(input_dim,))]
    for _ in range(num_layers - 1):
        ll.append(layers.Dense(units, activation='relu'))
    ll.append(layers.Dense(1))
    return keras.Sequential(ll)


# ============================================================
# CORE: run one seed (one dataset, one teacher config)
# ============================================================
def run_one_seed(data, seed, teacher_cfg_name='base',
                 teacher_type='ensemble', compute_cost=False):
    """
    Returns:
      results: {model_name: {alpha: {'val': mse, 'test': mse}}}
      meta: teacher stats + timing info
    """
    X_tr, X_va, X_te = data['X_tr'], data['X_va'], data['X_te']
    y_tr, y_va, y_te = data['y_tr'], data['y_va'], data['y_te']
    yt_tr = data.get('yt_tr')
    nf = data['n_features']

    cfg = TEACHER_CONFIGS[teacher_cfg_name]
    t0_teacher = time.time()
    pred_ens, pred_nn, pred_xgb, pred_gb = train_teacher_ensemble(data, cfg, seed)
    teacher_time = time.time() - t0_teacher

    # Select teacher type
    if teacher_type == 'ensemble':
        teacher_predict = pred_ens
    elif teacher_type == 'nn_only':
        teacher_predict = pred_nn
    elif teacher_type == 'xgb_only':
        teacher_predict = pred_xgb
    elif teacher_type == 'gb_only':
        teacher_predict = pred_gb
    else:
        teacher_predict = pred_ens

    t_train = teacher_predict(X_tr)
    t_test = teacher_predict(X_te)

    teacher_mse = mean_squared_error(y_te, t_test)

    # Denoising analysis (only for synthetic)
    noise_var, teacher_vs_true, denoising_pct = None, None, None
    if yt_tr is not None:
        noise_var = mean_squared_error(yt_tr, y_tr)
        teacher_vs_true = mean_squared_error(yt_tr, t_train)
        denoising_pct = (1 - teacher_vs_true / noise_var) * 100 if noise_var > 0 else 0

    results = {}
    timing = {}

    # ---- sklearn students ----
    students = get_sklearn_students(seed)
    for name, model in students.items():
        results[name] = {}
        t0 = time.time()
        for alpha in ALPHAS:
            m = clone(model)
            blended = alpha * y_tr + (1 - alpha) * t_train
            m.fit(X_tr, blended)
            results[name][alpha] = {
                'val':  mean_squared_error(y_va, m.predict(X_va)),
                'test': mean_squared_error(y_te, m.predict(X_te)),
            }
        timing[name] = time.time() - t0

    # ---- MLP students ----
    for name, (units, nl) in MLP_CONFIGS.items():
        results[name] = {}
        t0 = time.time()
        for alpha in ALPHAS:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            m = create_small_mlp(units, nl, nf)
            blended = alpha * y_tr + (1 - alpha) * t_train
            m.compile(optimizer='adam', loss='mse')
            m.fit(X_tr, blended, epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            results[name][alpha] = {
                'val':  mean_squared_error(y_va, m.predict(X_va, verbose=0).flatten()),
                'test': mean_squared_error(y_te, m.predict(X_te, verbose=0).flatten()),
            }
        timing[name] = time.time() - t0

    meta = {
        'teacher_mse': teacher_mse,
        'noise_var': noise_var,
        'teacher_vs_true': teacher_vs_true,
        'denoising_pct': denoising_pct,
        'teacher_time': teacher_time,
        'student_timing': timing,
    }
    return results, meta


# ============================================================
# BASELINE F: Label smoothing for MLPs (Experiment F)
# ============================================================
def run_label_smoothing_baselines(data, seed):
    """Train MLPs with Gaussian label smoothing (no teacher)."""
    X_tr, X_va, X_te = data['X_tr'], data['X_va'], data['X_te']
    y_tr, y_va, y_te = data['y_tr'], data['y_va'], data['y_te']
    nf = data['n_features']
    rng = np.random.RandomState(seed)

    results = {}
    for name, (units, nl) in MLP_CONFIGS.items():
        results[name] = {}
        # Baseline: standard training
        tf.random.set_seed(seed)
        m = create_small_mlp(units, nl, nf)
        m.compile(optimizer='adam', loss='mse')
        m.fit(X_tr, y_tr, epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
        results[name]['standard'] = mean_squared_error(y_te, m.predict(X_te, verbose=0).flatten())

        # Label smoothing: add Gaussian noise to labels
        for ls_sigma in LS_SIGMAS:
            tf.random.set_seed(seed)
            m = create_small_mlp(units, nl, nf)
            y_smooth = y_tr + ls_sigma * rng.randn(len(y_tr))
            m.compile(optimizer='adam', loss='mse')
            m.fit(X_tr, y_smooth, epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
            results[name][f'LS({ls_sigma})'] = mean_squared_error(y_te, m.predict(X_te, verbose=0).flatten())

        # Ridge-like: early stopping (50% epochs)
        tf.random.set_seed(seed)
        m = create_small_mlp(units, nl, nf)
        m.compile(optimizer='adam', loss='mse')
        m.fit(X_tr, y_tr, epochs=MLP_EPOCHS // 2, batch_size=MLP_BATCH, verbose=0)
        results[name]['EarlyStop'] = mean_squared_error(y_te, m.predict(X_te, verbose=0).flatten())

        # L2 regularised
        tf.random.set_seed(seed)
        ll = [layers.Dense(units, activation='relu', input_shape=(nf,),
                           kernel_regularizer=keras.regularizers.l2(0.01))]
        for _ in range(nl - 1):
            ll.append(layers.Dense(units, activation='relu',
                                   kernel_regularizer=keras.regularizers.l2(0.01)))
        ll.append(layers.Dense(1))
        m = keras.Sequential(ll)
        m.compile(optimizer='adam', loss='mse')
        m.fit(X_tr, y_tr, epochs=MLP_EPOCHS, batch_size=MLP_BATCH, verbose=0)
        results[name]['L2reg'] = mean_squared_error(y_te, m.predict(X_te, verbose=0).flatten())

    return results


# ============================================================
# AGGREGATION
# ============================================================
def aggregate_seeds(all_data):
    """Aggregate results across seeds. Alpha selected on val, reported on test."""
    model_names = list(all_data[0][0].keys())
    out = {}

    for name in model_names:
        base_tests, best_tests, best_alphas = [], [], []
        per_alpha_tests = {a: [] for a in ALPHAS}

        for results, meta in all_data:
            res = results[name]
            base_tests.append(res[1.0]['test'])
            best_a = min(ALPHAS, key=lambda a: res[a]['val'])
            best_tests.append(res[best_a]['test'])
            best_alphas.append(best_a)
            for a in ALPHAS:
                per_alpha_tests[a].append(res[a]['test'])

        base_arr = np.array(base_tests)
        best_arr = np.array(best_tests)
        delta = base_arr - best_arr  # positive = KD helped (lower MSE)

        n = len(delta)
        if n >= 10:
            try:
                stat, pval = stats.wilcoxon(delta, alternative='greater')
            except ValueError:
                stat, pval = 0, 1.0
            tname = 'Wilcoxon'
        else:
            stat, pval = stats.ttest_rel(base_arr, best_arr)
            tname = 'paired-t'

        # 95% CI for improvement
        imp_pct = (delta / base_arr) * 100
        if n > 1:
            ci = stats.t.interval(0.95, n-1, loc=imp_pct.mean(), scale=stats.sem(imp_pct))
        else:
            ci = (imp_pct.mean(), imp_pct.mean())

        out[name] = {
            'base_mean': base_arr.mean(), 'base_std': base_arr.std(),
            'best_mean': best_arr.mean(), 'best_std': best_arr.std(),
            'imp_mean': imp_pct.mean(), 'imp_std': imp_pct.std(),
            'imp_ci': ci,
            'wins': np.mean(delta > 0) * 100,
            'pval': pval, 'test_name': tname,
            'alpha_mode': float(stats.mode(best_alphas, keepdims=False).mode),
            'alpha_mean': np.mean(best_alphas),
            'per_alpha': {a: (np.mean(per_alpha_tests[a]), np.std(per_alpha_tests[a])) for a in ALPHAS},
            # Raw per-seed values (one entry per seed) for reproducibility / re-formatting.
            'per_seed': {
                'seeds': [42 + i for i in range(n)],
                'base_test': base_tests,
                'best_test': best_tests,
                'best_alpha': best_alphas,
                'imp_pct': imp_pct.tolist(),
                'per_alpha_test': {a: per_alpha_tests[a] for a in ALPHAS},
            },
        }

    # Teacher & timing
    t_mses = [m['teacher_mse'] for _, m in all_data]
    nvs = [m['noise_var'] for _, m in all_data if m['noise_var'] is not None]
    tvts = [m['teacher_vs_true'] for _, m in all_data if m['teacher_vs_true'] is not None]
    denoise = [m['denoising_pct'] for _, m in all_data if m['denoising_pct'] is not None]
    ttimes = [m['teacher_time'] for _, m in all_data]
    stimes = {}
    for name in model_names:
        stimes[name] = np.mean([m['student_timing'].get(name, 0) for _, m in all_data])

    out['_teacher'] = {
        'mse_mean': np.mean(t_mses), 'mse_std': np.std(t_mses),
        'noise_var': np.mean(nvs) if nvs else None,
        'tvt': np.mean(tvts) if tvts else None,
        'denoise': np.mean(denoise) if denoise else None,
        'train_time': np.mean(ttimes),
        'per_seed': {
            'seeds': [42 + i for i in range(len(t_mses))],
            'teacher_mse': t_mses,
            'noise_var': nvs if nvs else None,
            'teacher_vs_true': tvts if tvts else None,
            'denoising_pct': denoise if denoise else None,
        },
    }
    out['_timing'] = stimes
    return out


# ============================================================
# PRINTING
# ============================================================
def print_table(agg, title=""):
    if title:
        print(f"\n{title}")
    print(f"{'Model':<12} {'Base MSE':<22} {'Distil MSE':<22} {'Improv%':<30} {'Wins%':<7} {'p-val':<10} {'a*':<5}")
    print("-" * 110)
    models = [k for k in agg if not k.startswith('_')]
    for name in sorted(models, key=lambda n: agg[n]['best_mean']):
        a = agg[name]
        sig = '***' if a['pval'] < 0.001 else ('**' if a['pval'] < 0.01 else ('*' if a['pval'] < 0.05 else ''))
        print(f"{name:<12} {a['base_mean']:7.4f} +/- {a['base_std']:.4f}  "
              f"{a['best_mean']:7.4f} +/- {a['best_std']:.4f}  "
              f"{a['imp_mean']:+8.3f}% [{a['imp_ci'][0]:+.3f},{a['imp_ci'][1]:+.3f}]  "
              f"{a['wins']:5.0f}%  {a['pval']:.4f}{sig} {a['alpha_mode']:.1f}")


def print_per_alpha_table(agg, title=""):
    if title:
        print(f"\n{title}")
    print(f"{'Model':<12}", end="")
    for a in ALPHAS:
        print(f" {'a='+str(a):<16}", end="")
    print(f" {'Best a':<6}")
    print("-" * (12 + 17 * len(ALPHAS) + 8))
    models = [k for k in agg if not k.startswith('_')]
    for name in sorted(models, key=lambda n: agg[n]['best_mean']):
        d = agg[name]
        print(f"{name:<12}", end="")
        for a in ALPHAS:
            m, s = d['per_alpha'][a]
            print(f" {m:7.4f}+/-{s:.3f} ", end="")
        print(f" {d['alpha_mode']:.1f}")


def print_category_summary(agg, title=""):
    if title:
        print(f"\n{title}")
    models = {k: v for k, v in agg.items() if not k.startswith('_')}
    cats = {
        'KNN':   [n for n in models if 'KNN' in n],
        'MLP':   [n for n in models if 'MLP' in n],
        'Trees': [n for n in models if any(x in n for x in ['RF', 'XGB', 'GB', 'DT'])],
        'SVR':   [n for n in models if 'SVR' in n],
    }
    print(f"{'Category':<10} {'Avg a*':<8} {'Avg Improv%':<14} {'Avg Wins%':<10}")
    print("-" * 44)
    for cat, names in cats.items():
        if names:
            aa = np.mean([models[n]['alpha_mode'] for n in names])
            ai = np.mean([models[n]['imp_mean'] for n in names])
            aw = np.mean([models[n]['wins'] for n in names])
            print(f"{cat:<10} {aa:<8.2f} {ai:<14.3f} {aw:<10.0f}")


def print_timing(agg, title=""):
    if title:
        print(f"\n{title}")
    t = agg['_teacher']
    print(f"  Teacher training time: {t['train_time']:.1f}s (mean)")
    print(f"  Student training times (mean over alpha sweep, seconds):")
    for name, stime in sorted(agg['_timing'].items(), key=lambda x: x[1]):
        print(f"    {name:<12} {stime:.2f}s")


# ============================================================
# EXPERIMENT A: Main multi-seed
# ============================================================
def experiment_A():
    print("\n" + "=" * 80)
    print(f"EXPERIMENT A: MAIN MULTI-SEED (seeds={N_SEEDS}, sigma={BASE_SIGMA})")
    print("Test set NEVER used for alpha selection.")
    print("=" * 80)

    all_data = []
    for s in range(N_SEEDS):
        seed = 42 + s
        t0 = time.time()
        print(f"  Seed {s+1}/{N_SEEDS} (seed={seed})...", end=" ", flush=True)
        data = generate_synthetic(seed, BASE_SIGMA)
        results, meta = run_one_seed(data, seed, 'base', 'ensemble', compute_cost=True)
        print(f"done ({time.time()-t0:.0f}s)")
        all_data.append((results, meta))

    agg = aggregate_seeds(all_data)

    # Teacher
    t = agg['_teacher']
    print(f"\n--- Teacher Performance (ensemble, {N_SEEDS} seeds) ---")
    print(f"  Test MSE:        {t['mse_mean']:.4f} +/- {t['mse_std']:.4f}")
    if t['noise_var'] is not None:
        print(f"  Noise variance:  {t['noise_var']:.4f}")
        print(f"  Teacher vs true: {t['tvt']:.4f}")
        print(f"  Noise removed:   {t['denoise']:.1f}%")

    print_per_alpha_table(agg, "--- Per-Alpha MSE Table (mean +/- std across seeds) ---")
    print_table(agg, "--- Distillation Benefit (val-selected alpha vs baseline a=1.0) ---")
    print_category_summary(agg, "--- Category Summary ---")
    print_timing(agg, "--- Computational Cost ---")

    models = {k: v for k, v in agg.items() if not k.startswith('_')}
    helped = sum(1 for v in models.values() if v['imp_mean'] > 0)
    avg_imp = np.mean([v['imp_mean'] for v in models.values()])
    print(f"\nOverall: {helped}/{len(models)} improved, avg improvement: {avg_imp:+.3f}%")

    return agg, all_data


# ============================================================
# EXPERIMENT B: Noise sweep
# ============================================================
def experiment_B():
    print("\n" + "=" * 80)
    print(f"EXPERIMENT B: NOISE SWEEP (sigmas={SIGMA_GRID}, seeds={N_SEEDS})")
    print("=" * 80)

    sweep = {}
    for sigma in SIGMA_GRID:
        print(f"\n--- sigma = {sigma} ---")
        all_data = []
        for s in range(N_SEEDS):
            seed = 42 + s
            print(f"  Seed {s+1}/{N_SEEDS}...", end=" ", flush=True)
            t0 = time.time()
            data = generate_synthetic(seed, sigma)
            results, meta = run_one_seed(data, seed, 'base', 'ensemble')
            all_data.append((results, meta))
            print(f"({time.time()-t0:.0f}s)", end="")
        print()
        sweep[sigma] = aggregate_seeds(all_data)

    # Summary table
    print("\n" + "-" * 80)
    print("NOISE SWEEP SUMMARY")
    print(f"{'sigma':<7} {'TeachMSE':<11} {'Denoise%':<10} {'Helped':<10} {'AvgImprov%':<12} {'MLP Improv%':<12}")
    print("-" * 64)
    for sigma in SIGMA_GRID:
        a = sweep[sigma]; t = a['_teacher']
        models = {k: v for k, v in a.items() if not k.startswith('_')}
        h = sum(1 for v in models.values() if v['imp_mean'] > 0)
        ai = np.mean([v['imp_mean'] for v in models.values()])
        mi = np.mean([v['imp_mean'] for k, v in models.items() if 'MLP' in k])
        dn = f"{t['denoise']:.1f}" if t['denoise'] is not None else "N/A"
        print(f"{sigma:<7.2f} {t['mse_mean']:<11.4f} {dn:<10} {h}/{len(models):<7} {ai:<12.3f} {mi:<12.3f}")

    # Per-model improvement vs sigma
    rep = ['KNN(k5)', 'MLP(16x2)', 'MLP(8x2)', 'GB(sm)', 'DT(d3)']
    print(f"\n{'Model':<12}", end="")
    for sigma in SIGMA_GRID:
        print(f" sig={sigma:<6}", end="")
    print()
    print("-" * (12 + 10 * len(SIGMA_GRID)))
    for mn in rep:
        print(f"{mn:<12}", end="")
        for sigma in SIGMA_GRID:
            if mn in sweep[sigma]:
                print(f" {sweep[sigma][mn]['imp_mean']:>+8.3f}%", end="")
            else:
                print(f" {'N/A':>7}", end="")
        print()

    # Optimal alpha vs sigma
    print(f"\n{'Model':<12}", end="")
    for sigma in SIGMA_GRID:
        print(f" a*@{sigma:<6}", end="")
    print()
    print("-" * (12 + 10 * len(SIGMA_GRID)))
    for mn in rep:
        print(f"{mn:<12}", end="")
        for sigma in SIGMA_GRID:
            if mn in sweep[sigma]:
                print(f" {sweep[sigma][mn]['alpha_mode']:>6.1f}", end="")
            else:
                print(f" {'N/A':>6}", end="")
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
            data = generate_synthetic(seed, BASE_SIGMA)
            results, meta = run_one_seed(data, seed, tname, 'ensemble')
            all_data.append((results, meta))
            print(f"({time.time()-t0:.0f}s)", end="")
        print()
        ablation[tname] = aggregate_seeds(all_data)

    print("\n" + "-" * 80)
    print("TEACHER ABLATION SUMMARY")
    print(f"{'Teacher':<8} {'TeachMSE':<12} {'Denoise%':<10} {'Helped':<10} {'AvgImp%':<12} {'MLP Imp%':<12}")
    print("-" * 66)
    for tname in ['weak', 'base', 'strong']:
        a = ablation[tname]; t = a['_teacher']
        models = {k: v for k, v in a.items() if not k.startswith('_')}
        h = sum(1 for v in models.values() if v['imp_mean'] > 0)
        ai = np.mean([v['imp_mean'] for v in models.values()])
        mi = np.mean([v['imp_mean'] for k, v in models.items() if 'MLP' in k])
        dn = f"{t['denoise']:.1f}" if t['denoise'] is not None else "N/A"
        print(f"{tname:<8} {t['mse_mean']:<12.4f} {dn:<10} {h}/{len(models):<7} {ai:<12.3f} {mi:<12.3f}")

    # Per-model
    rep = ['KNN(k5)', 'MLP(16x2)', 'MLP(8x2)', 'GB(sm)', 'SVR(rbf)']
    print(f"\n{'Model':<12} {'Weak Imp%':<14} {'Base Imp%':<14} {'Strong Imp%':<14}")
    print("-" * 56)
    for mn in rep:
        print(f"{mn:<12}", end="")
        for tname in ['weak', 'base', 'strong']:
            if mn in ablation[tname]:
                print(f" {ablation[tname][mn]['imp_mean']:>+8.3f}%     ", end="")
            else:
                print(f" {'N/A':>7}      ", end="")
        print()

    # Alpha shift
    print(f"\n{'Model':<12} {'Weak a*':<12} {'Base a*':<12} {'Strong a*':<12}")
    print("-" * 48)
    for mn in rep:
        print(f"{mn:<12}", end="")
        for tname in ['weak', 'base', 'strong']:
            if mn in ablation[tname]:
                print(f" {ablation[tname][mn]['alpha_mode']:>6.1f}     ", end="")
            else:
                print(f" {'N/A':>6}      ", end="")
        print()

    return ablation


# ============================================================
# EXPERIMENT D: Single vs ensemble teacher
# ============================================================
def experiment_D():
    print("\n" + "=" * 80)
    print(f"EXPERIMENT D: SINGLE VS ENSEMBLE TEACHER (seeds={N_SEEDS})")
    print("=" * 80)

    teacher_types = ['nn_only', 'xgb_only', 'gb_only', 'ensemble']
    ablation = {}
    for ttype in teacher_types:
        print(f"\n--- Teacher type: {ttype} ---")
        all_data = []
        for s in range(N_SEEDS):
            seed = 42 + s
            print(f"  Seed {s+1}/{N_SEEDS}...", end=" ", flush=True)
            t0 = time.time()
            data = generate_synthetic(seed, BASE_SIGMA)
            results, meta = run_one_seed(data, seed, 'base', ttype)
            all_data.append((results, meta))
            print(f"({time.time()-t0:.0f}s)", end="")
        print()
        ablation[ttype] = aggregate_seeds(all_data)

    print("\n" + "-" * 80)
    print("SINGLE VS ENSEMBLE TEACHER SUMMARY")
    print(f"{'TeacherType':<12} {'TeachMSE':<12} {'Helped':<10} {'AvgImp%':<12} {'MLP Imp%':<12}")
    print("-" * 58)
    for ttype in teacher_types:
        a = ablation[ttype]; t = a['_teacher']
        models = {k: v for k, v in a.items() if not k.startswith('_')}
        h = sum(1 for v in models.values() if v['imp_mean'] > 0)
        ai = np.mean([v['imp_mean'] for v in models.values()])
        mi = np.mean([v['imp_mean'] for k, v in models.items() if 'MLP' in k])
        print(f"{ttype:<12} {t['mse_mean']:<12.4f} {h}/{len(models):<7} {ai:<12.3f} {mi:<12.3f}")

    return ablation


# ============================================================
# EXPERIMENT E: Real datasets
# ============================================================
def experiment_E():
    print("\n" + "=" * 80)
    print(f"EXPERIMENT E: REAL DATASETS (seeds={N_SEEDS})")
    print("=" * 80)

    datasets = ['california', 'diabetes', 'friedman']
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

        print_table(agg, f"--- {dsname}: Distillation Results ---")

    # Cross-dataset summary
    print("\n" + "-" * 80)
    print("REAL DATASETS CROSS-SUMMARY")
    print(f"{'Dataset':<14} {'TeachMSE':<12} {'Helped':<10} {'AvgImp%':<12} {'MLP Imp%':<12} {'KNN Imp%':<12}")
    print("-" * 72)
    for dsname in datasets:
        a = real_results[dsname]; t = a['_teacher']
        models = {k: v for k, v in a.items() if not k.startswith('_')}
        h = sum(1 for v in models.values() if v['imp_mean'] > 0)
        ai = np.mean([v['imp_mean'] for v in models.values()])
        mi = np.mean([v['imp_mean'] for k, v in models.items() if 'MLP' in k])
        ki = np.mean([v['imp_mean'] for k, v in models.items() if 'KNN' in k])
        print(f"{dsname:<14} {t['mse_mean']:<12.4f} {h}/{len(models):<7} {ai:<12.3f} {mi:<12.3f} {ki:<12.3f}")

    return real_results


# ============================================================
# EXPERIMENT F: Baselines (label smoothing, L2, early stop)
# ============================================================
def experiment_F(all_data_A):
    print("\n" + "=" * 80)
    print(f"EXPERIMENT F: BASELINES vs KD FOR MLPs (seeds={N_SEEDS})")
    print("=" * 80)

    baseline_results = {}
    for s in range(N_SEEDS):
        seed = 42 + s
        print(f"  Seed {s+1}/{N_SEEDS}...", end=" ", flush=True)
        t0 = time.time()
        data = generate_synthetic(seed, BASE_SIGMA)
        bl = run_label_smoothing_baselines(data, seed)
        baseline_results[seed] = bl
        print(f"({time.time()-t0:.0f}s)")

    # Aggregate: for each MLP, compare standard vs LS vs EarlyStop vs L2 vs KD
    print(f"\n{'MLP':<12} {'Standard':<14} {'BestLS':<14} {'EarlyStop':<14} {'L2reg':<14} {'BestKD':<14}")
    print("-" * 82)
    for name in MLP_CONFIGS:
        std_vals = [baseline_results[42+s][name]['standard'] for s in range(N_SEEDS)]
        es_vals = [baseline_results[42+s][name]['EarlyStop'] for s in range(N_SEEDS)]
        l2_vals = [baseline_results[42+s][name]['L2reg'] for s in range(N_SEEDS)]

        # Best label smoothing per seed
        best_ls_vals = []
        for s in range(N_SEEDS):
            ls_mses = {k: v for k, v in baseline_results[42+s][name].items()
                       if k.startswith('LS(')}
            best_ls_vals.append(min(ls_mses.values()))

        # Best KD from experiment A
        kd_vals = []
        for results, meta in all_data_A:
            res = results[name]
            best_a = min(ALPHAS, key=lambda a: res[a]['val'])
            kd_vals.append(res[best_a]['test'])

        print(f"{name:<12} "
              f"{np.mean(std_vals):7.4f}+/-{np.std(std_vals):.3f} "
              f"{np.mean(best_ls_vals):7.4f}+/-{np.std(best_ls_vals):.3f} "
              f"{np.mean(es_vals):7.4f}+/-{np.std(es_vals):.3f} "
              f"{np.mean(l2_vals):7.4f}+/-{np.std(l2_vals):.3f} "
              f"{np.mean(kd_vals):7.4f}+/-{np.std(kd_vals):.3f}")

    return baseline_results


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    t_start = time.time()
    print("=" * 80)
    print("KNOWLEDGE DISTILLATION FOR REGRESSION - ENHANCED EXPERIMENTS")
    print(f"Mode: {MODE} | Seeds: {N_SEEDS} | Sigmas: {SIGMA_GRID}")
    print("=" * 80)

    agg_A, all_data_A = experiment_A()
    sweep_B = experiment_B()
    ablation_C = experiment_C()
    single_D = experiment_D()
    real_E = experiment_E()
    baselines_F = experiment_F(all_data_A)

    # Persist full-precision results (including raw per-seed values) so any table
    # can be reformatted or any statistic recomputed without re-running.
    def _to_native(obj):
        """Recursively convert numpy types/containers to JSON-native Python."""
        if isinstance(obj, dict):
            return {(_to_native(k) if not isinstance(k, str) else k): _to_native(v)
                    for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_native(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    def _clean_agg(agg):
        """Drop unpicklable callables; keep stats + per_seed arrays."""
        out = {}
        for k, v in agg.items():
            if k == '_timing':
                out[k] = _to_native(v)
            elif k == '_teacher':
                out[k] = _to_native(v)
            elif k.startswith('_'):
                out[k] = _to_native(v)
            else:
                out[k] = _to_native(v)
        return out

    try:
        payload = {
            'config': {
                'mode': MODE, 'n_seeds': N_SEEDS, 'seeds': [42 + i for i in range(N_SEEDS)],
                'base_sigma': BASE_SIGMA, 'alphas': ALPHAS, 'sigma_grid': SIGMA_GRID,
            },
            'experiment_A': _clean_agg(agg_A),
            'experiment_B': {str(sigma): _clean_agg(sweep_B[sigma]) for sigma in sweep_B},
            'experiment_C': {tname: _clean_agg(ablation_C[tname]) for tname in ablation_C},
            'experiment_D': {ttype: _clean_agg(single_D[ttype]) for ttype in single_D},
            'experiment_E': {ds: _clean_agg(real_E[ds]) for ds in real_E},
            # Experiment F is keyed by seed -> model -> {strategy: test_mse}; already per-seed.
            'experiment_F': _to_native(baselines_F),
        }
        with open('results_regression_full.json', 'w') as f:
            json.dump(payload, f, indent=2)
        print("\nFull-precision results (with per-seed values) written to "
              "results_regression_full.json")
    except Exception as e:
        print(f"\n[warn] could not write JSON dump: {e}")

    # Flat per-seed CSV (one row per experiment x condition x model x seed).
    # Easy to load with pandas: pd.read_csv('results_regression_per_seed.csv')
    def _emit_agg_rows(writer, experiment, condition, agg):
        for model, v in agg.items():
            if model.startswith('_'):
                continue
            ps = v.get('per_seed')
            if not ps:
                continue
            seeds = ps['seeds']
            for i, seed in enumerate(seeds):
                row = {
                    'experiment': experiment,
                    'condition': condition,
                    'model': model,
                    'seed': seed,
                    'base_test_mse': ps['base_test'][i],
                    'best_test_mse': ps['best_test'][i],
                    'best_alpha': ps['best_alpha'][i],
                    'improvement_pct': ps['imp_pct'][i],
                }
                # one column per alpha's test MSE
                for a in ALPHAS:
                    row[f'test_mse_a{a}'] = ps['per_alpha_test'][a][i]
                writer.writerow(row)

    try:
        fieldnames = (['experiment', 'condition', 'model', 'seed',
                       'base_test_mse', 'best_test_mse', 'best_alpha', 'improvement_pct']
                      + [f'test_mse_a{a}' for a in ALPHAS])
        with open('results_regression_per_seed.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            _emit_agg_rows(w, 'A', 'main', agg_A)
            for sigma in sweep_B:
                _emit_agg_rows(w, 'B', f'sigma={sigma}', sweep_B[sigma])
            for tname in ablation_C:
                _emit_agg_rows(w, 'C', f'teacher={tname}', ablation_C[tname])
            for ttype in single_D:
                _emit_agg_rows(w, 'D', f'teacher={ttype}', single_D[ttype])
            for ds in real_E:
                _emit_agg_rows(w, 'E', f'dataset={ds}', real_E[ds])
        print("Per-seed model results written to results_regression_per_seed.csv")
    except Exception as e:
        print(f"[warn] could not write per-seed CSV: {e}")

    # Experiment F has a different shape (seed -> model -> {strategy: test_mse}).
    try:
        # discover strategy columns from the first record
        any_seed = next(iter(baselines_F))
        any_model = next(iter(baselines_F[any_seed]))
        strategies = list(baselines_F[any_seed][any_model].keys())
        f_fields = ['experiment', 'model', 'seed'] + strategies
        with open('results_regression_baselines_per_seed.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=f_fields)
            w.writeheader()
            for seed, models in baselines_F.items():
                for model, strat_mses in models.items():
                    row = {'experiment': 'F', 'model': model, 'seed': seed}
                    for strat in strategies:
                        row[strat] = strat_mses.get(strat)
                    w.writerow(row)
        print("Per-seed baseline (Exp F) results written to "
              "results_regression_baselines_per_seed.csv")
    except Exception as e:
        print(f"[warn] could not write baselines CSV: {e}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"ALL REGRESSION EXPERIMENTS COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'=' * 80}")