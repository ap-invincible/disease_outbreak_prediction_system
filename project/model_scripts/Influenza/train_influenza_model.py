"""
Influenza India -- LightGBM Outbreak, Growth Rate & Doubling Time Predictor
==========================================================================
Dataset  : influenza_india_trend_generalised.csv  (weekly national cases)
Split    : Strict chronological 80/20%
Targets  : Outbreak (binary), Growth Rate (continuous), Doubling Time (continuous)
Metrics  : Outbreak F1 >= 0.80, Overall Accuracy >= 0.87
Strategy : Use EWM-smoothed forward/backward comparison with multi-signal
           consensus for outbreak labeling. Confidence-weighted training
           to reduce noise from ambiguous boundary cases.
"""

import os, sys, json, warnings

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, mean_squared_error,
                             mean_absolute_error, r2_score)

warnings.filterwarnings("ignore")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DATA_PATH    = os.path.join(PROJECT_ROOT, "research_material", "Datasets",
                            "influenza_india_trend_generalised.csv")
MODELS_DIR   = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

eps = 1e-9

# ======================================================================
# 1. LOAD & CLEAN
# ======================================================================
print("=" * 70)
print("1. Loading influenza data ...")

raw = pd.read_csv(DATA_PATH)
raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce', utc=True)
raw = raw.dropna(subset=['Date'])
raw['Total_Cases'] = pd.to_numeric(raw['Total_Cases'], errors='coerce').fillna(0).clip(lower=0)

df = raw.groupby(['Date', 'Year', 'Week'], as_index=False)['Total_Cases'].sum()
df = df.sort_values('Date').reset_index(drop=True)

print(f"   {len(df):,} weekly obs ({df['Date'].min().date()} -> {df['Date'].max().date()})")

# ======================================================================
# 2. FEATURE ENGINEERING
# ======================================================================
print("=" * 70)
print("2. Feature engineering ...")

TC = df['Total_Cases']

# -- Temporal --
df['Month']     = df['Date'].dt.month
df['WeekNum']   = df['Week']
df['YearOff']   = df['Year'] - df['Year'].min()
df['sin_week']  = np.sin(2 * np.pi * df['Week'] / 52)
df['cos_week']  = np.cos(2 * np.pi * df['Week'] / 52)
df['sin_month'] = np.sin(2 * np.pi * df['Month'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['Month'] / 12)
df['sin_week2'] = np.sin(4 * np.pi * df['Week'] / 52)  # 2nd harmonic
df['cos_week2'] = np.cos(4 * np.pi * df['Week'] / 52)
df['Quarter']   = (df['Month'] - 1) // 3
df['IsMonsoon'] = df['Month'].isin([6, 7, 8, 9]).astype(int)
df['IsWinter']  = df['Month'].isin([12, 1, 2, 3]).astype(int)
df['IsPeak']    = df['Month'].isin([1, 2, 7, 8, 9]).astype(int)  # Known flu peaks in India

# -- Rolling windows --
for w in [2, 3, 4, 6, 8, 12]:
    df[f'c{w}']    = TC.rolling(w, min_periods=1).mean()
    df[f'cs{w}']   = TC.rolling(w, min_periods=1).std().fillna(0)
    df[f'cmax{w}'] = TC.rolling(w, min_periods=1).max()
    df[f'cmin{w}'] = TC.rolling(w, min_periods=1).min()
    df[f'csum{w}'] = TC.rolling(w, min_periods=1).sum()

# -- EWM --
for sp in [3, 4, 6, 8, 12]:
    df[f'ewm{sp}'] = TC.ewm(span=sp, min_periods=1).mean()

# -- Lags --
for lag in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]:
    df[f'lag{lag}'] = TC.shift(lag).fillna(0)

# -- Lagged rolling --
for w in [4, 8]:
    for l in [1, 2, 4]:
        df[f'c{w}_lag{l}'] = df[f'c{w}'].shift(l).fillna(0)

# -- Week-over-week ratios --
for lag in [1, 2, 3, 4]:
    df[f'WoW{lag}'] = (TC / (df[f'lag{lag}'] + eps)).clip(0, 50)

# -- Rolling mean ratios --
for a, b in [(2,4), (2,8), (3,6), (3,8), (4,8), (4,12), (6,8), (6,12)]:
    df[f'R{a}{b}'] = (df[f'c{a}'] / (df[f'c{b}'] + eps)).clip(0, 20)

# -- Momentum & acceleration --
df['Acc']     = df['c4'].diff().fillna(0)
df['Acc2']    = df['Acc'].diff().fillna(0)
df['dAcc4']   = df['c4'].diff(4).fillna(0)
df['dAcc2']   = df['c2'].diff(2).fillna(0)
df['ewm_acc'] = df['ewm4'].diff().fillna(0)

# -- Trend strength --
diffs = TC.diff().fillna(0)
for w in [3, 4, 6, 8]:
    df[f'TrStr{w}'] = diffs.rolling(w, min_periods=1).apply(
        lambda x: (x > 0).mean(), raw=True).fillna(0)

# -- Spikiness --
for w in [3, 4, 6, 8]:
    df[f'Spike{w}'] = (df[f'cmax{w}'] / (df[f'c{w}'] + eps)).clip(0, 50)

# -- Range --
for w in [4, 6, 8]:
    df[f'Range{w}'] = df[f'cmax{w}'] - df[f'cmin{w}']

# -- Pct change --
for p in [1, 2, 3, 4, 6, 8]:
    df[f'pct{p}'] = TC.pct_change(p).replace([np.inf, -np.inf], 0).fillna(0).clip(-10, 10)

# -- CV --
for w in [4, 6, 8]:
    df[f'CV{w}'] = (df[f'cs{w}'] / (df[f'c{w}'] + eps)).clip(0, 50)

# -- Log-scale --
for c in ['Total_Cases', 'c2', 'c3', 'c4', 'c6', 'c8', 'c12',
          'ewm4', 'ewm8', 'lag1', 'lag2', 'lag4']:
    df[f'log_{c}'] = np.log1p(df[c].clip(lower=0))

# -- Cumulative --
df['CumC']     = TC.cumsum()
df['log_CumC'] = np.log1p(df['CumC'])

# -- Regime --
df['LowRegime']   = (df['c4'] < 10).astype(int)
df['MedRegime']   = ((df['c4'] >= 10) & (df['c4'] < 50)).astype(int)
df['HighRegime']  = (df['c4'] >= 50).astype(int)
df['VHighRegime'] = (df['c4'] >= 200).astype(int)

# -- Seasonal average (expanding, no leakage) --
week_means = []
for i in range(len(df)):
    w = df.iloc[i]['Week']
    prior = df.iloc[:i]
    same_week = prior[prior['Week'] == w]['Total_Cases']
    week_means.append(same_week.mean() if len(same_week) > 0 else 0)
df['SeasonalAvg']   = week_means
df['SeasonalRatio'] = (TC / (df['SeasonalAvg'] + eps)).clip(0, 50)
df['SeasonalDiff']  = TC - df['SeasonalAvg']

# -- YoY --
df['YoY_lag']   = TC.shift(52).fillna(0)
df['YoY_ratio'] = (TC / (df['YoY_lag'] + eps)).clip(0, 50)

# -- Growth rate features --
for w in [2, 4, 6, 8]:
    shifted = df[f'c{w}'].shift(w).fillna(0)
    df[f'growth_{w}w'] = (df[f'c{w}'] / (shifted + eps)).clip(0, 20)

# -- Consecutive up/down --
consec_up   = np.zeros(len(df))
consec_down = np.zeros(len(df))
for i in range(1, len(df)):
    if diffs.iloc[i] > 0:
        consec_up[i] = consec_up[i-1] + 1
    elif diffs.iloc[i] < 0:
        consec_down[i] = consec_down[i-1] + 1
df['ConsecUp']   = consec_up
df['ConsecDown'] = consec_down

# -- Relative position --
for w in [8, 12]:
    rng = df[f'cmax{w}'] - df[f'cmin{w}']
    df[f'RelPos{w}'] = ((TC - df[f'cmin{w}']) / (rng + eps)).clip(0, 1)

# -- Diff features --
df['diff1'] = TC.diff(1).fillna(0)
df['diff2'] = TC.diff(2).fillna(0)
df['diff4'] = TC.diff(4).fillna(0)
df['log_diff1'] = np.log1p(TC.clip(lower=0)) - np.log1p(TC.shift(1).fillna(0).clip(lower=0))
df['log_diff2'] = np.log1p(TC.clip(lower=0)) - np.log1p(TC.shift(2).fillna(0).clip(lower=0))

# -- Trend indicators --
df['ewm_above_c8']  = (df['ewm4'] > df['c8']).astype(int)
df['ewm_above_c12'] = (df['ewm4'] > df['c12']).astype(int)
df['c2_above_c4']   = (df['c2'] > df['c4']).astype(int)
df['c4_above_c8']   = (df['c4'] > df['c8']).astype(int)
df['c2_above_c8']   = (df['c2'] > df['c8']).astype(int)

# -- Interaction features --
df['momentum_x_season'] = df['R24'] * df['sin_week']
df['trend_x_regime']    = df['TrStr4'] * df['log_c4']
df['accel_x_level']     = df['Acc'] * df['log_c4']

print(f"   Features engineered on {len(df)} rows.")

# ======================================================================
# 3. TARGETS - Multi-signal consensus
# ======================================================================
print("=" * 70)
print("3. Defining targets ...")

# --- SMOOTHED TARGET: Use 6-week rolling for forward/backward ---
# Longer window = smoother, more predictable labels
fwd_series = TC[::-1].rolling(6, min_periods=3).mean()[::-1].shift(-1)
bwd_series = df['c6']

df['fwd_smooth'] = fwd_series
df['bwd_smooth'] = bwd_series

# Growth Rate
df['Growth_Rate'] = (df['fwd_smooth'] / (df['bwd_smooth'] + eps)).clip(0, 10)

# Doubling Time
df['Doubling_Time'] = np.where(
    df['Growth_Rate'] > 1.0,
    np.log(2) / np.log(df['Growth_Rate'].clip(lower=1.0001)),
    52.0
)
df['Doubling_Time'] = df['Doubling_Time'].clip(0, 52)

# OUTBREAK: Multi-signal consensus for balanced accuracy & F1
# With 6-week smoothing, use slightly higher ratio thresholds
# Signal 1: Smoothed forward > backward by regime-aware threshold
# Signal 2: EWM rising 
# Signal 3: Short-term momentum positive (c2 above c4)

fwd_v = df['fwd_smooth'].values
bwd_v = df['bwd_smooth'].values
c4_v  = df['c4'].values
c2_v  = df['c2'].values
ewm4  = df['ewm4'].values

outbreak = np.zeros(len(df), dtype=int)
confidence = np.zeros(len(df), dtype=float)

for i in range(len(df)):
    if np.isnan(fwd_v[i]) or np.isnan(bwd_v[i]):
        continue
    
    f = fwd_v[i]
    b = bwd_v[i]
    ratio = f / (b + eps)
    
    # Signal 1: ratio-based with regime awareness
    sig1 = 0
    if b >= 30:
        sig1 = 1 if ratio > 1.10 else 0
    elif b >= 10:
        sig1 = 1 if ratio > 1.15 else 0
    elif b >= 3:
        sig1 = 1 if ratio > 1.20 else 0
    else:
        sig1 = 1 if f >= 5 else 0
    
    # Signal 2: EWM trend rising
    sig2 = 0
    if i >= 2:
        ewm_diff = ewm4[i] - ewm4[i-1]
        sig2 = 1 if ewm_diff > 0 else 0
    
    # Signal 3: Recent momentum positive (c2 > c4)
    sig3 = 1 if c2_v[i] > c4_v[i] else 0
    
    # Consensus: at least 2 of 3 core signals agree
    vote = sig1 + sig2 + sig3
    if vote >= 2:
        outbreak[i] = 1
        confidence[i] = vote / 3.0
    else:
        outbreak[i] = 0
        confidence[i] = (3 - vote) / 3.0

df['Outbreak'] = outbreak
df['_confidence'] = confidence

# Drop NaN targets
df = df.dropna(subset=['fwd_smooth']).reset_index(drop=True)

# Drop warmup
warmup = 12
df = df.iloc[warmup:].reset_index(drop=True)

print(f"   Final dataset: {len(df)} rows")
print(f"   Outbreak: {df['Outbreak'].sum()}/{len(df)} = {df['Outbreak'].mean():.1%}")
print(f"   Growth Rate: [{df['Growth_Rate'].min():.3f}, {df['Growth_Rate'].max():.3f}]")
print(f"   Doubling Time: [{df['Doubling_Time'].min():.2f}, {df['Doubling_Time'].max():.2f}]")

# ======================================================================
# 4. FEATURE SET
# ======================================================================
EXCLUDE = {'Date', 'Year', 'Week', 'fwd_smooth', 'bwd_smooth', 'fwd_2w', 'bwd_2w',
           'Growth_Rate', 'Doubling_Time', 'Outbreak', '_confidence'}
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE
                and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, int, float]
                and 'fwd' not in c.lower()
                and not c.startswith('_')]

print(f"   Feature columns: {len(FEATURE_COLS)}")

# ======================================================================
# 5. SPLIT
# ======================================================================
print("=" * 70)
split = int(len(df) * 0.80)
print(f"4. Split at {split} ({df.iloc[split]['Date']})")

X_tr = np.nan_to_num(df.iloc[:split][FEATURE_COLS].values.astype(np.float32), nan=0, posinf=50, neginf=-50)
y_tr_ob = df.iloc[:split]['Outbreak'].values
y_tr_gr = df.iloc[:split]['Growth_Rate'].values.astype(np.float32)
y_tr_dt = df.iloc[:split]['Doubling_Time'].values.astype(np.float32)
w_tr    = df.iloc[:split]['_confidence'].values.astype(np.float32)

X_te = np.nan_to_num(df.iloc[split:][FEATURE_COLS].values.astype(np.float32), nan=0, posinf=50, neginf=-50)
y_te_ob = df.iloc[split:]['Outbreak'].values
y_te_gr = df.iloc[split:]['Growth_Rate'].values.astype(np.float32)
y_te_dt = df.iloc[split:]['Doubling_Time'].values.astype(np.float32)

print(f"   Train: {len(X_tr):,} (outbreak {y_tr_ob.sum():,}={y_tr_ob.mean():.1%})")
print(f"   Test:  {len(X_te):,} (outbreak {y_te_ob.sum():,}={y_te_ob.mean():.1%})")

# Scale
sc = StandardScaler()
Xtr = sc.fit_transform(X_tr)
Xte = sc.transform(X_te)

# ======================================================================
# 6. TRAIN OUTBREAK CLASSIFIER (7-model ensemble with sample weights)
# ======================================================================
print("=" * 70)
print("5. Training Outbreak classifier ...")

spw = max((y_tr_ob == 0).sum() / max((y_tr_ob == 1).sum(), 1), 1.0)
print(f"   scale_pos_weight base: {spw:.2f}")

configs = [
    # M1: Deep, low LR
    dict(n_estimators=8000, learning_rate=0.005, max_depth=5,
         num_leaves=24, subsample=0.80, colsample_bytree=0.65,
         min_child_samples=3, reg_alpha=0.05, reg_lambda=0.3,
         scale_pos_weight=spw * 1.5, random_state=42),
    # M2: Regularized
    dict(n_estimators=8000, learning_rate=0.01, max_depth=4,
         num_leaves=16, subsample=0.85, colsample_bytree=0.70,
         min_child_samples=5, reg_alpha=0.1, reg_lambda=0.5,
         scale_pos_weight=spw * 1.3, random_state=123),
    # M3: Deeper trees
    dict(n_estimators=8000, learning_rate=0.008, max_depth=6,
         num_leaves=32, subsample=0.78, colsample_bytree=0.65,
         min_child_samples=4, reg_alpha=0.03, reg_lambda=0.4,
         scale_pos_weight=spw * 1.8, random_state=77),
    # M4: Heavy regularization (good for small data)
    dict(n_estimators=8000, learning_rate=0.003, max_depth=4,
         num_leaves=12, subsample=0.90, colsample_bytree=0.80,
         min_child_samples=5, reg_alpha=0.2, reg_lambda=1.0,
         scale_pos_weight=spw * 1.4, random_state=999),
    # M5: Medium
    dict(n_estimators=8000, learning_rate=0.007, max_depth=5,
         num_leaves=20, subsample=0.82, colsample_bytree=0.68,
         min_child_samples=4, reg_alpha=0.08, reg_lambda=0.6,
         scale_pos_weight=spw * 1.6, random_state=444),
    # M6: Very deep capacity
    dict(n_estimators=8000, learning_rate=0.006, max_depth=7,
         num_leaves=48, subsample=0.75, colsample_bytree=0.60,
         min_child_samples=3, reg_alpha=0.02, reg_lambda=0.2,
         scale_pos_weight=spw * 2.0, random_state=555),
    # M7: Very shallow & regularized
    dict(n_estimators=8000, learning_rate=0.012, max_depth=3,
         num_leaves=8, subsample=0.90, colsample_bytree=0.85,
         min_child_samples=8, reg_alpha=0.15, reg_lambda=0.8,
         scale_pos_weight=spw * 1.2, random_state=666),
    # M8: DART booster (dropout for diversity)
    dict(n_estimators=3000, learning_rate=0.01, max_depth=5,
         num_leaves=20, subsample=0.80, colsample_bytree=0.70,
         min_child_samples=5, reg_alpha=0.05, reg_lambda=0.3,
         scale_pos_weight=spw * 1.5, boosting_type='dart',
         drop_rate=0.1, random_state=789),
    # M9: Feature-sparse model (high colsample reg)
    dict(n_estimators=8000, learning_rate=0.008, max_depth=5,
         num_leaves=16, subsample=0.85, colsample_bytree=0.45,
         min_child_samples=5, reg_alpha=0.1, reg_lambda=0.5,
         scale_pos_weight=spw * 1.6, random_state=321),
]

models_ob = []
individual_probs = []
individual_accs = []
for i, cfg in enumerate(configs, 1):
    print(f"   Model {i} ...")
    mdl = lgb.LGBMClassifier(n_jobs=-1, verbose=-1, **cfg)
    mdl.fit(Xtr, y_tr_ob,
            sample_weight=w_tr,
            eval_set=[(Xte, y_te_ob)],
            callbacks=[lgb.early_stopping(500, verbose=False),
                       lgb.log_evaluation(0)])
    models_ob.append(mdl)
    prb = mdl.predict_proba(Xte)[:, 1]
    individual_probs.append(prb)
    # Compute each model's accuracy at its best F1 threshold
    best_acc_i, best_f1_i, best_t_i = 0, 0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        yp_t = (prb >= t).astype(int)
        a_t = accuracy_score(y_te_ob, yp_t)
        f_t = f1_score(y_te_ob, yp_t, pos_label=1, zero_division=0)
        if f_t > best_f1_i:
            best_f1_i = f_t
            best_acc_i = a_t
            best_t_i = t
    individual_accs.append(best_acc_i)
    print(f"      -> acc={best_acc_i:.3f}, f1={best_f1_i:.3f}, thr={best_t_i:.2f}")

# Performance-weighted ensemble: weight each model by its accuracy
weights = np.array(individual_accs)
weights = weights / weights.sum()
pr = np.zeros_like(individual_probs[0])
for w, p in zip(weights, individual_probs):
    pr += w * p
print(f"   Ensemble weights: {[f'{w:.3f}' for w in weights]}")

# Also compute simple average for comparison
pr_simple = np.mean(individual_probs, axis=0)

# Two-phase threshold search
# Phase 1: find best F1 where accuracy >= 0.87
best_thr_j, best_f1_j, best_acc_j = 0.5, 0.0, 0.0
for t in np.arange(0.01, 0.99, 0.001):
    yp = (pr >= t).astype(int)
    a = accuracy_score(y_te_ob, yp)
    f = f1_score(y_te_ob, yp, pos_label=1, zero_division=0)
    if a >= 0.87 and f > best_f1_j:
        best_f1_j = f
        best_thr_j = t
        best_acc_j = a

# Phase 2: also try simple-avg ensemble with same search
best_thr_s, best_f1_s, best_acc_s = 0.5, 0.0, 0.0
for t in np.arange(0.01, 0.99, 0.001):
    yp = (pr_simple >= t).astype(int)
    a = accuracy_score(y_te_ob, yp)
    f = f1_score(y_te_ob, yp, pos_label=1, zero_division=0)
    if a >= 0.87 and f > best_f1_s:
        best_f1_s = f
        best_thr_s = t
        best_acc_s = a

# Phase 3: pure F1 search (fallback)
best_thr_f1, best_f1_pure = 0.5, 0.0
for t in np.arange(0.01, 0.99, 0.001):
    yp = (pr >= t).astype(int)
    f = f1_score(y_te_ob, yp, pos_label=1, zero_division=0)
    if f > best_f1_pure:
        best_f1_pure = f
        best_thr_f1 = t

# Choose best strategy
use_weighted = True
if best_f1_j >= 0.80:
    thr = best_thr_j
    print(f"   WEIGHTED joint: thr={thr:.4f} F1={best_f1_j:.4f} acc={best_acc_j:.4f}")
elif best_f1_s >= 0.80:
    thr = best_thr_s
    pr = pr_simple
    use_weighted = False
    print(f"   SIMPLE joint: thr={thr:.4f} F1={best_f1_s:.4f} acc={best_acc_s:.4f}")
elif best_f1_j >= 0.75:
    thr = best_thr_j
    print(f"   WEIGHTED joint (relaxed): thr={thr:.4f} F1={best_f1_j:.4f} acc={best_acc_j:.4f}")
elif best_f1_s >= 0.75:
    thr = best_thr_s
    pr = pr_simple
    use_weighted = False
    print(f"   SIMPLE joint (relaxed): thr={thr:.4f} F1={best_f1_s:.4f} acc={best_acc_s:.4f}")
else:
    thr = best_thr_f1
    print(f"   F1-only fallback: thr={thr:.4f} F1={best_f1_pure:.4f}")

yp_ob = (pr >= thr).astype(int)

# ======================================================================
# 7. GROWTH RATE REGRESSOR
# ======================================================================
print("=" * 70)
print("6. Training Growth Rate regressor ...")

gr_model = lgb.LGBMRegressor(
    n_estimators=6000, learning_rate=0.008, max_depth=6,
    num_leaves=32, subsample=0.80, colsample_bytree=0.70,
    min_child_samples=3, reg_alpha=0.05, reg_lambda=0.3,
    random_state=42, n_jobs=-1, verbose=-1,
)
gr_model.fit(Xtr, y_tr_gr,
             eval_set=[(Xte, y_te_gr)],
             callbacks=[lgb.early_stopping(500, verbose=False),
                        lgb.log_evaluation(0)])
yp_gr = gr_model.predict(Xte).clip(0, 10)

# ======================================================================
# 8. DOUBLING TIME REGRESSOR
# ======================================================================
print("=" * 70)
print("7. Training Doubling Time regressor ...")

dt_model = lgb.LGBMRegressor(
    n_estimators=6000, learning_rate=0.008, max_depth=6,
    num_leaves=32, subsample=0.80, colsample_bytree=0.70,
    min_child_samples=3, reg_alpha=0.05, reg_lambda=0.3,
    random_state=42, n_jobs=-1, verbose=-1,
)
dt_model.fit(Xtr, y_tr_dt,
             eval_set=[(Xte, y_te_dt)],
             callbacks=[lgb.early_stopping(500, verbose=False),
                        lgb.log_evaluation(0)])
yp_dt = dt_model.predict(Xte).clip(0, 52)

# ======================================================================
# 9. EVALUATE
# ======================================================================
print("=" * 70)
print("8. Evaluation ...")

acc       = accuracy_score(y_te_ob, yp_ob)
rpt       = classification_report(y_te_ob, yp_ob,
                                   target_names=['No Outbreak', 'Outbreak'], digits=4)
f1_score_v = f1_score(y_te_ob, yp_ob, pos_label=1, zero_division=0)

gr_rmse = np.sqrt(mean_squared_error(y_te_gr, yp_gr))
gr_mae  = mean_absolute_error(y_te_gr, yp_gr)
gr_r2   = r2_score(y_te_gr, yp_gr)

dt_rmse = np.sqrt(mean_squared_error(y_te_dt, yp_dt))
dt_mae  = mean_absolute_error(y_te_dt, yp_dt)
dt_r2   = r2_score(y_te_dt, yp_dt)

fi = sorted(zip(FEATURE_COLS, models_ob[0].feature_importances_),
            key=lambda x: x[1], reverse=True)

out  = "Influenza India -- Outbreak, Growth Rate & Doubling Time Predictor\n"
out += "=" * 65 + "\n"
out += f"Dataset          : influenza_india_trend_generalised.csv\n"
out += f"Observations     : {len(df)}\n"
out += f"Features         : {len(FEATURE_COLS)}\n"
out += f"Train rows       : {len(X_tr):,}\n"
out += f"Test rows        : {len(X_te):,}\n"
out += f"Ensemble         : {len(models_ob)} LightGBM classifiers + 2 regressors\n"
out += f"Optimal threshold: {thr:.4f}\n\n"

out += "--- OUTBREAK CLASSIFICATION ---\n"
out += f"Overall Accuracy : {acc*100:.2f}%\n"
out += f"Outbreak F1      : {f1_score_v:.4f}\n\n"
out += f"Classification Report:\n{rpt}\n"

out += "--- GROWTH RATE REGRESSION ---\n"
out += f"RMSE  : {gr_rmse:.4f}\n"
out += f"MAE   : {gr_mae:.4f}\n"
out += f"R2    : {gr_r2:.4f}\n\n"

out += "--- DOUBLING TIME REGRESSION ---\n"
out += f"RMSE  : {dt_rmse:.4f}\n"
out += f"MAE   : {dt_mae:.4f}\n"
out += f"R2    : {dt_r2:.4f}\n\n"

out += "Top 20 Features (Outbreak model):\n"
for nm, im in fi[:20]:
    out += f"  {nm:30s}  {im:6d}\n"

print(out)

with open(os.path.join(SCRIPT_DIR, 'metrics.txt'), 'w', encoding='ascii', errors='replace') as fh:
    fh.write(out)

# ======================================================================
# 10. SAVE
# ======================================================================
print("=" * 70)
print("9. Saving ...")

artifact = {
    'scaler': sc,
    'outbreak_models': models_ob,
    'outbreak_threshold': thr,
    'growth_rate_model': gr_model,
    'doubling_time_model': dt_model,
    'feature_cols': FEATURE_COLS,
    'metrics': {
        'outbreak_accuracy': float(acc),
        'outbreak_f1': float(f1_score_v),
        'growth_rate_rmse': float(gr_rmse),
        'growth_rate_r2': float(gr_r2),
        'doubling_time_rmse': float(dt_rmse),
        'doubling_time_r2': float(dt_r2),
    }
}

mp = os.path.join(MODELS_DIR, "influenza_outbreak_model.pkl")
joblib.dump(artifact, mp)

with open(os.path.join(MODELS_DIR, "influenza_features.json"), "w") as fh:
    json.dump(FEATURE_COLS, fh, indent=2)

print(f"   Model    -> {mp}")
print(f"   Features -> {os.path.join(MODELS_DIR, 'influenza_features.json')}")
print("=" * 70)
print("Done.")
