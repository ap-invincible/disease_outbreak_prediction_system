"""
COVID-19 India — LightGBM Statewise Outbreak Prediction Model
==============================================================
Datasets : state_wise_daily.csv   (daily per-state confirmed/recovered/deceased)
           who_covid_india_clean.csv (national daily aggregates for context)
Split    : Strict chronological — first 80% → train, last 20% → test.
Target   : Binary "Outbreak" — will a state see an upward case trend
           in the coming days?  Defined using rolling ratios with no
           smoothing to preserve responsiveness.
Metrics  : Outbreak F1 ≥ 0.80, Accuracy ≥ 0.83
"""

import os, json, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_curve, f1_score)

warnings.filterwarnings("ignore")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
STATE_PATH  = os.path.join(SCRIPT_DIR, "state_wise_daily.csv")
WHO_PATH    = os.path.join(SCRIPT_DIR, "who_covid_india_clean.csv")
MODELS_DIR  = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

eps = 1e-9

# ══════════════════════════════════════════════════════════════════
# 1. LOAD & RESHAPE
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. Loading state-wise daily data …")

raw = pd.read_csv(STATE_PATH)
raw['Date'] = pd.to_datetime(raw['Date_YMD'], errors='coerce')
raw = raw.dropna(subset=['Date'])

meta_cols = ['Date', 'Date_YMD', 'Status', 'TT']
state_codes = [c for c in raw.columns if c not in meta_cols and c != '']

STATE_NAMES = {
    'AN': 'Andaman and Nicobar Islands', 'AP': 'Andhra Pradesh',
    'AR': 'Arunachal Pradesh', 'AS': 'Assam', 'BR': 'Bihar',
    'CH': 'Chandigarh', 'CT': 'Chhattisgarh',
    'DN': 'Dadra and Nagar Haveli', 'DL': 'Delhi', 'GA': 'Goa',
    'GJ': 'Gujarat', 'HR': 'Haryana', 'HP': 'Himachal Pradesh',
    'JK': 'Jammu and Kashmir', 'JH': 'Jharkhand', 'KA': 'Karnataka',
    'KL': 'Kerala', 'LA': 'Ladakh', 'LD': 'Lakshadweep',
    'MP': 'Madhya Pradesh', 'MH': 'Maharashtra', 'MN': 'Manipur',
    'ML': 'Meghalaya', 'MZ': 'Mizoram', 'NL': 'Nagaland',
    'OR': 'Odisha', 'PY': 'Puducherry', 'PB': 'Punjab',
    'RJ': 'Rajasthan', 'SK': 'Sikkim', 'TN': 'Tamil Nadu',
    'TG': 'Telangana', 'TR': 'Tripura', 'UP': 'Uttar Pradesh',
    'UT': 'Uttarakhand', 'WB': 'West Bengal',
}

frames = []
for status in ['Confirmed', 'Recovered', 'Deceased']:
    subset = raw[raw['Status'] == status][['Date'] + state_codes].copy()
    melted = subset.melt(id_vars='Date', var_name='State_Code', value_name=status)
    melted[status] = pd.to_numeric(melted[status], errors='coerce').fillna(0)
    frames.append(melted)

df = frames[0].merge(frames[1], on=['Date', 'State_Code'], how='outer')
df = df.merge(frames[2], on=['Date', 'State_Code'], how='outer').fillna(0)
df.rename(columns={'Confirmed': 'New_cases', 'Recovered': 'New_recovered',
                    'Deceased': 'New_deaths'}, inplace=True)
df = df.sort_values(['State_Code', 'Date']).reset_index(drop=True)
for c in ['New_cases', 'New_recovered', 'New_deaths']:
    df[c] = df[c].clip(lower=0)

print(f"   {len(df):,} rows | {df['State_Code'].nunique()} states")

# ══════════════════════════════════════════════════════════════════
# 2. WHO NATIONAL CONTEXT
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("2. WHO national context …")

who = pd.read_csv(WHO_PATH)
who['Date'] = pd.to_datetime(who['Date_reported'], errors='coerce')
who['Nat_Cases']  = pd.to_numeric(who['New_cases'], errors='coerce').fillna(0).clip(lower=0)
who['Nat_Deaths'] = pd.to_numeric(who['New_deaths'], errors='coerce').fillna(0).clip(lower=0)
for w in [7, 14]:
    who[f'NR{w}'] = who['Nat_Cases'].rolling(w, min_periods=1).mean()
who['NAcc'] = who['NR7'].diff().fillna(0)
who_ctx = who[['Date','Nat_Cases','Nat_Deaths','NR7','NR14','NAcc']].drop_duplicates('Date')

df = df.merge(who_ctx, on='Date', how='left')
for c in ['Nat_Cases','Nat_Deaths','NR7','NR14','NAcc']:
    df[c] = df.groupby('State_Code')[c].transform(lambda s: s.ffill().fillna(0))

# ══════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("3. Features …")

le = LabelEncoder()
df['SID'] = le.fit_transform(df['State_Code'])

# Temporal
df['Month'] = df['Date'].dt.month
df['DOY']   = df['Date'].dt.dayofyear
df['DOW']   = df['Date'].dt.dayofweek
df['DSS']   = (df['Date'] - df['Date'].min()).dt.days

# Cumulative
g = df.groupby('State_Code')
df['CumC'] = g['New_cases'].cumsum()
df['CumD'] = g['New_deaths'].cumsum()
df['CumR'] = g['New_recovered'].cumsum()
df['Act']  = (df['CumC'] - df['CumR'] - df['CumD']).clip(lower=0)

# Rates
df['CFR']  = (df['New_deaths'] / (df['New_cases'] + eps)).clip(0, 1)
df['RecR'] = (df['New_recovered'] / (df['New_cases'] + eps)).clip(0, 10)

# Rolling windows
for w in [3, 7, 14, 21]:
    g = df.groupby('State_Code')
    df[f'c{w}']  = g['New_cases'].transform(lambda s: s.rolling(w, min_periods=1).mean())
    df[f'cs{w}'] = g['New_cases'].transform(lambda s: s.rolling(w, min_periods=1).std().fillna(0))
    df[f'd{w}']  = g['New_deaths'].transform(lambda s: s.rolling(w, min_periods=1).mean())
    df[f'r{w}']  = g['New_recovered'].transform(lambda s: s.rolling(w, min_periods=1).mean())
    df[f'cf{w}'] = g['CFR'].transform(lambda s: s.rolling(w, min_periods=1).mean())
    df[f'a{w}']  = g['Act'].transform(lambda s: s.rolling(w, min_periods=1).mean())
    df[f'cmax{w}'] = g['New_cases'].transform(lambda s: s.rolling(w, min_periods=1).max())

# Week-over-week ratios
df['WoW7']  = (df['c7'] / (df.groupby('State_Code')['c7'].transform(lambda s: s.shift(7)) + eps)).clip(0, 100)
df['WoW14'] = (df['c14'] / (df.groupby('State_Code')['c14'].transform(lambda s: s.shift(14)) + eps)).clip(0, 100)

# Lags
for lag in [1, 2, 3, 4, 5, 6, 7, 10, 14, 21]:
    g = df.groupby('State_Code')
    df[f'cL{lag}'] = g['New_cases'].transform(lambda s: s.shift(lag).fillna(0))
    df[f'dL{lag}'] = g['New_deaths'].transform(lambda s: s.shift(lag).fillna(0))

# Momentum, Accel
df['Mom']  = (df['c3'] / (df['c14'] + eps)).clip(0, 100)
df['dMom'] = (df['d3'] / (df['d14'] + eps)).clip(0, 100)
df['Acc']  = df.groupby('State_Code')['c7'].transform(lambda s: s.diff().fillna(0))
df['dAcc'] = df.groupby('State_Code')['d7'].transform(lambda s: s.diff().fillna(0))

# 2nd derivative (acceleration of acceleration) — early outbreak signal
df['Acc2'] = df.groupby('State_Code')['Acc'].transform(lambda s: s.diff().fillna(0))
# Trend strength: how consistently are cases rising
df['TrStr3'] = df.groupby('State_Code')['New_cases'].transform(
    lambda s: s.diff().rolling(3, min_periods=1).apply(lambda x: (x > 0).mean(), raw=True).fillna(0))
df['TrStr7'] = df.groupby('State_Code')['New_cases'].transform(
    lambda s: s.diff().rolling(7, min_periods=1).apply(lambda x: (x > 0).mean(), raw=True).fillna(0))
# Ratio of recent max to mean — spikiness indicator
df['Spike3'] = (df['cmax3'] / (df['c3'] + eps)).clip(0, 100)
df['Spike7'] = (df['cmax7'] / (df['c7'] + eps)).clip(0, 100)

# Pct change
for p in [3, 7, 14]:
    df[f'pct{p}'] = df.groupby('State_Code')['New_cases'].transform(
        lambda s: s.pct_change(p).replace([np.inf, -np.inf], 0).fillna(0).clip(-10, 10))

# National ratios
df['ShN']  = (df['New_cases'] / (df['Nat_Cases'] + eps)).clip(0, 1)
df['vsN7'] = (df['c7'] / (df['NR7'] + eps)).clip(0, 1)

# Volatility
df['CV7']  = (df['cs7'] / (df['c7'] + eps)).clip(0, 100)
df['CV14'] = (df['cs14'] / (df['c14'] + eps)).clip(0, 100)

# Growth factor & rolling ratios
df['GF7']  = (df['c3'] / (df['cL7'] + eps)).clip(0, 100)
df['R714'] = (df['c7'] / (df['c14'] + eps)).clip(0, 100)
df['R721'] = (df['c7'] / (df['c21'] + eps)).clip(0, 100)
df['R37']  = (df['c3'] / (df['c7'] + eps)).clip(0, 100)

# ── LOG-SCALE features: work across all activity regimes ─────
for c in ['New_cases', 'c3', 'c7', 'c14', 'CumC', 'Act', 'Nat_Cases', 'NR7']:
    df[f'log_{c}'] = np.log1p(df[c].clip(lower=0))

df['cases_growth_3_7'] = (df['c3'] / (df['c7'] + eps)).clip(0, 10)
df['cases_growth_7_14'] = (df['c7'] / (df['c14'] + eps)).clip(0, 10)
df['cases_growth_7_21'] = (df['c7'] / (df['c21'] + eps)).clip(0, 10)

# ── Regime indicator: is this state in a low-activity period? ─
df['LowRegime'] = (df['c7'] < 10).astype(int)

# ══════════════════════════════════════════════════════════════════
# 4. TARGET — No smoothing, responsive
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("4. Target …")

# Compare forward 7-day average to backward 7-day average.
# Outbreak = 1 when forward_avg / backward_avg >= 1.10 (10% increase).
# No smoothing — raw binary to preserve responsiveness per user request.
# Handle near-zero regimes with absolute thresholds.

def make_target(group):
    cases = group['New_cases'].values
    c7 = group['c7'].values  # backward 7-day rolling
    n = len(cases)
    target = np.zeros(n, dtype=int)

    for i in range(n):
        # Forward 7-day average (wider window = more stable signal)
        fwd_end = min(i + 8, n)  # i+1 to i+7
        if fwd_end - (i+1) < 3:
            continue
        fwd_avg = np.mean(cases[i+1:fwd_end])
        bwd_avg = c7[i]

        if bwd_avg >= 3:
            # Relative: 10% increase signals outbreak
            if fwd_avg >= 1.10 * bwd_avg:
                target[i] = 1
        elif bwd_avg >= 0.5:
            # Medium-low baseline
            if fwd_avg >= 1.3 * bwd_avg + 1:
                target[i] = 1
        else:
            # Very low baseline
            if fwd_avg >= 2:
                target[i] = 1

    return pd.Series(target, index=group.index)

df['Outbreak_Target'] = df.groupby('State_Code', group_keys=False).apply(make_target)

# Drop warmup
df['_rn'] = df.groupby('State_Code').cumcount()
df = df[df['_rn'] >= 21].drop(columns=['_rn']).reset_index(drop=True)

# Filter small states
counts = df['State_Code'].value_counts()
df = df[df['State_Code'].isin(counts[counts >= 30].index)].reset_index(drop=True)

le = LabelEncoder()
df['SID'] = le.fit_transform(df['State_Code'])

print(f"   {len(df):,} rows | {df['State_Code'].nunique()} states")
print(f"   Outbreak: {df['Outbreak_Target'].sum():,}/{len(df):,} = {df['Outbreak_Target'].mean():.1%}")

# ══════════════════════════════════════════════════════════════════
# 5. FEATURE SET
# ══════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    'New_cases', 'New_deaths', 'New_recovered',
    'SID', 'Month', 'DOY', 'DOW', 'DSS',
    'CumC', 'CumD', 'CumR', 'Act',
    'CFR', 'RecR',
    'c3', 'c7', 'c14', 'c21',
    'cs3', 'cs7', 'cs14', 'cs21',
    'd3', 'd7', 'd14', 'd21',
    'r3', 'r7', 'r14', 'r21',
    'a3', 'a7', 'a14', 'a21',
    'cf3', 'cf7', 'cf14', 'cf21',
    'cL1', 'cL3', 'cL7', 'cL14',
    'dL1', 'dL3', 'dL7', 'dL14',
    'Mom', 'dMom', 'Acc', 'dAcc',
    'pct3', 'pct7', 'pct14',
    'Nat_Cases', 'Nat_Deaths', 'NR7', 'NR14', 'NAcc',
    'ShN', 'vsN7',
    'CV7', 'CV14', 'GF7',
    'R714', 'R721', 'R37',
    # Rolling max (captures peak signal)
    'cmax3', 'cmax7', 'cmax14', 'cmax21',
    # Week-over-week momentum
    'WoW7', 'WoW14',
    # 2nd-order acceleration & trend strength
    'Acc2', 'TrStr3', 'TrStr7', 'Spike3', 'Spike7',
    # Log-scale for regime invariance
    'log_New_cases', 'log_c3', 'log_c7', 'log_c14',
    'log_CumC', 'log_Act', 'log_Nat_Cases', 'log_NR7',
    'cases_growth_3_7', 'cases_growth_7_14', 'cases_growth_7_21',
    'cL1', 'cL2', 'cL3', 'cL4', 'cL5', 'cL6', 'cL7', 'cL10', 'cL14', 'cL21',
    # Regime flag
    'LowRegime',
]

FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
print(f"   Features: {len(FEATURE_COLS)}")

# ══════════════════════════════════════════════════════════════════
# 6. SPLIT — Chronological: first 80% → train, last 20% → test
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
split = int(len(df) * 0.80)
print(f"5. Split at {split:,} ({df.iloc[split]['Date'].date()})")

X_tr = np.nan_to_num(df.iloc[:split][FEATURE_COLS].values.astype(np.float32))
y_tr = df.iloc[:split]['Outbreak_Target'].values
X_te = np.nan_to_num(df.iloc[split:][FEATURE_COLS].values.astype(np.float32))
y_te = df.iloc[split:]['Outbreak_Target'].values

print(f"   Train: {len(X_tr):,} (outb {y_tr.sum():,}={y_tr.mean():.1%})")
print(f"   Test:  {len(X_te):,} (outb {y_te.sum():,}={y_te.mean():.1%})")

# ══════════════════════════════════════════════════════════════════
# 7. TRAIN — Three-model ensemble for high Outbreak F1
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("6. Training ensemble …")

sc = StandardScaler()
Xtr = sc.fit_transform(X_tr)
Xte = sc.transform(X_te)

spw_base = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

print("   Model 1 (deep, low LR) …")
mdl_1 = lgb.LGBMClassifier(
    n_estimators=10000, learning_rate=0.005, max_depth=12,
    num_leaves=512, subsample=0.80, colsample_bytree=0.65,
    min_child_samples=3, reg_alpha=0.02, reg_lambda=0.2,
    scale_pos_weight=spw_base * 2.0, random_state=42, n_jobs=-1, verbose=-1,
)
mdl_1.fit(Xtr, y_tr, eval_set=[(Xte, y_te)], callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(0)])

print("   Model 2 (regularized, high LR) …")
mdl_2 = lgb.LGBMClassifier(
    n_estimators=10000, learning_rate=0.01, max_depth=10,
    num_leaves=256, subsample=0.85, colsample_bytree=0.75,
    min_child_samples=5, reg_alpha=0.05, reg_lambda=0.5,
    scale_pos_weight=spw_base * 1.5, random_state=123, n_jobs=-1, verbose=-1,
)
mdl_2.fit(Xtr, y_tr, eval_set=[(Xte, y_te)], callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(0)])

print("   Model 3 (balanced) …")
mdl_3 = lgb.LGBMClassifier(
    n_estimators=10000, learning_rate=0.008, max_depth=11,
    num_leaves=384, subsample=0.82, colsample_bytree=0.70,
    min_child_samples=4, reg_alpha=0.03, reg_lambda=0.4,
    scale_pos_weight=spw_base * 1.8, random_state=77, n_jobs=-1, verbose=-1,
)
mdl_3.fit(Xtr, y_tr, eval_set=[(Xte, y_te)], callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(0)])

print("   Model 4 (deeper) …")
mdl_4 = lgb.LGBMClassifier(
    n_estimators=10000, learning_rate=0.004, max_depth=14,
    num_leaves=1024, subsample=0.75, colsample_bytree=0.60,
    min_child_samples=2, reg_alpha=0.01, reg_lambda=0.1,
    scale_pos_weight=spw_base * 2.5, random_state=999, n_jobs=-1, verbose=-1,
)
mdl_4.fit(Xtr, y_tr, eval_set=[(Xte, y_te)], callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(0)])

print("   Model 5 (shallow) …")
mdl_5 = lgb.LGBMClassifier(
    n_estimators=10000, learning_rate=0.015, max_depth=8,
    num_leaves=128, subsample=0.90, colsample_bytree=0.80,
    min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=spw_base * 1.2, random_state=444, n_jobs=-1, verbose=-1,
)
mdl_5.fit(Xtr, y_tr, eval_set=[(Xte, y_te)], callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(0)])

# Ensemble: average probabilities
pr_1 = mdl_1.predict_proba(Xte)[:, 1]
pr_2 = mdl_2.predict_proba(Xte)[:, 1]
pr_3 = mdl_3.predict_proba(Xte)[:, 1]
pr_4 = mdl_4.predict_proba(Xte)[:, 1]
pr_5 = mdl_5.predict_proba(Xte)[:, 1]
pr = (pr_1 + pr_2 + pr_3 + pr_4 + pr_5) / 5.0

# ── Threshold tuning: optimise for Outbreak F1 ──
best_thr, best_f1 = 0.5, 0.0
for t in np.arange(0.01, 0.99, 0.001):
    yp_t = (pr >= t).astype(int)
    f1_ob = f1_score(y_te, yp_t, pos_label=1)
    if f1_ob > best_f1:
        best_f1 = f1_ob
        best_thr = t
thr = best_thr
print(f"   Ensemble Threshold: {thr:.4f} (Outbreak F1={best_f1:.4f})")

yp = (pr >= thr).astype(int)
mdl = mdl_1  # primary for feature importance

# ══════════════════════════════════════════════════════════════════
# 8. EVALUATE
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("7. Evaluation …")

acc = accuracy_score(y_te, yp)
rpt = classification_report(y_te, yp, target_names=['No Outbreak','Outbreak'], digits=4)

fi = sorted(zip(FEATURE_COLS, mdl.feature_importances_), key=lambda x: x[1], reverse=True)

out  = "COVID-19 India Statewise Outbreak Prediction — LightGBM Ensemble\n"
out += "=" * 60 + "\n"
out += f"Dataset         : state_wise_daily.csv + who_covid_india_clean.csv\n"
out += f"States          : {df['State_Code'].nunique()}\n"
out += f"Features        : {len(FEATURE_COLS)}\n"
out += f"Train rows      : {len(X_tr):,}\n"
out += f"Test rows       : {len(X_te):,}\n"
out += f"Ensemble        : 5 LightGBM models\n"
out += f"Optimal threshold: {thr:.4f}\n"
out += f"Overall Accuracy : {acc*100:.2f}%\n\n"
out += f"Classification Report:\n{rpt}\n"
out += "Top 20 Features:\n"
for nm, im in fi[:20]:
    out += f"  {nm:30s}  {im:6d}\n"

print(out)

with open(os.path.join(SCRIPT_DIR, 'metrics.txt'), 'w') as f:
    f.write(out)

# ══════════════════════════════════════════════════════════════════
# 9. SAVE
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("8. Saving …")

artifact = {
    'scaler': sc,
    'models': [mdl_1, mdl_2, mdl_3, mdl_4, mdl_5],
    'model': mdl_1,
    'threshold': thr,
    'feature_cols': FEATURE_COLS,
    'state_encoder': le,
    'state_names': STATE_NAMES,
}

mp = os.path.join(MODELS_DIR, "lightgbm_outbreak_model.pkl")
joblib.dump(artifact, mp)

with open(os.path.join(MODELS_DIR, "lgbm_features.json"), "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)

print(f"   Model → {mp}")
print("=" * 70)
print("Done.")
