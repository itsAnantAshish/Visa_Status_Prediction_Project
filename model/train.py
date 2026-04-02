import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ─── 1. LOAD DATA ────────────────────────────────────────────────
df = pd.read_csv("data/visa_dataset.csv", encoding='latin1')
print(f"Raw data shape: {df.shape}")
# ─── 2. BASIC CLEANING ───────────────────────────────────────────
# Drop rows with too many nulls
df.dropna(thresh=len(df.columns) - 2, inplace=True)

# Keep only relevant columns (adjust if your CSV has different names)
cols_needed = ['CASE_STATUS', 'EMPLOYER_NAME', 'JOB_TITLE',
               'FULL_TIME_POSITION', 'PREVAILING_WAGE', 'YEAR', 'WORKSITE']

# Only keep columns that actually exist
cols_needed = [c for c in cols_needed if c in df.columns]
df = df[cols_needed].copy()

# ─── 3. HANDLE MISSING VALUES ────────────────────────────────────
df['PREVAILING_WAGE'].fillna(df['PREVAILING_WAGE'].median(), inplace=True)
df['FULL_TIME_POSITION'].fillna('Y', inplace=True)
df['YEAR'].fillna(df['YEAR'].mode()[0], inplace=True)

for col in ['CASE_STATUS', 'EMPLOYER_NAME', 'JOB_TITLE', 'WORKSITE']:
    if col in df.columns:
        df[col].fillna('UNKNOWN', inplace=True)

# ─── 4. FEATURE ENGINEERING ──────────────────────────────────────
# Extract STATE from WORKSITE (reduces high cardinality)
if 'WORKSITE' in df.columns:
    df['STATE'] = df['WORKSITE'].apply(
        lambda x: str(x).split(',')[-1].strip()[:2].upper()
        if isinstance(x, str) and ',' in x else 'UNKNOWN'
    )
    df.drop('WORKSITE', axis=1, inplace=True)

# Wage category buckets
if 'PREVAILING_WAGE' in df.columns:
    df['WAGE_CATEGORY'] = pd.cut(
        df['PREVAILING_WAGE'],
        bins=[0, 40000, 70000, 100000, 150000, float('inf')],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    ).astype(str)

# Full time as binary
if 'FULL_TIME_POSITION' in df.columns:
    df['FULL_TIME_BINARY'] = (df['FULL_TIME_POSITION'].str.upper() == 'Y').astype(int)
    df.drop('FULL_TIME_POSITION', axis=1, inplace=True)

# ─── 5. SYNTHETIC TARGET VARIABLE ────────────────────────────────
# Since real processing dates aren't in dataset, we create realistic synthetic ones
np.random.seed(42)
base_time = 60

# Adjust based on wage (higher wage = faster processing typically)
wage_factor = np.where(
    df['PREVAILING_WAGE'] > 100000, -15,
    np.where(df['PREVAILING_WAGE'] > 70000, -5,
    np.where(df['PREVAILING_WAGE'] < 40000, 10, 0))
)

# Adjust based on case status
status_factor = np.where(
    df['CASE_STATUS'].str.upper().str.contains('CERTIFIED', na=False), -10,
    np.where(df['CASE_STATUS'].str.upper().str.contains('DENIED', na=False), 20, 0)
)

# Adjust based on year (more recent = slightly faster)
year_factor = np.where(df['YEAR'] >= 2018, -5,
              np.where(df['YEAR'] <= 2012, 10, 0))

noise = np.random.randint(-10, 10, len(df))

df['processing_time_days'] = np.clip(
    base_time + wage_factor + status_factor + year_factor + noise,
    15, 120
).astype(int)

print(f"\nTarget variable stats:")
print(df['processing_time_days'].describe())

# ─── 6. ENCODE CATEGORICALS ──────────────────────────────────────
cat_cols = ['CASE_STATUS', 'WAGE_CATEGORY']
if 'STATE' in df.columns:
    cat_cols.append('STATE')

# For high-cardinality columns, use frequency encoding
high_card_cols = ['EMPLOYER_NAME', 'JOB_TITLE']
for col in high_card_cols:
    if col in df.columns:
        freq = df[col].value_counts(normalize=True)
        df[col + '_FREQ'] = df[col].map(freq)
        df.drop(col, axis=1, inplace=True)

# Label encode remaining categoricals
encoders = {}
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

print(f"\nFinal columns: {df.columns.tolist()}")
print(f"Final shape: {df.shape}")

# ─── 7. TRAIN / TEST SPLIT ───────────────────────────────────────
X = df.drop('processing_time_days', axis=1)
y = df['processing_time_days']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")

# ─── 8. TRAIN XGBOOST ────────────────────────────────────────────
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# ─── 9. EVALUATE ─────────────────────────────────────────────────
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\n{'='*40}")
print(f"MAE  : {mae:.2f} days")
print(f"RMSE : {rmse:.2f} days")
print(f"R²   : {r2:.4f}")
print(f"{'='*40}")

# ─── 10. SAVE MODEL & METADATA ───────────────────────────────────
os.makedirs("model", exist_ok=True)

model_data = {
    'model': model,
    'encoders': encoders,
    'feature_columns': X.columns.tolist(),
    'metrics': {'mae': mae, 'rmse': rmse, 'r2': r2},
    'feature_importances': dict(zip(X.columns, model.feature_importances_))
}

joblib.dump(model_data, 'model/visa_model.pkl')
print("\n✅ Model saved to model/visa_model.pkl")
print(f"Feature columns: {X.columns.tolist()}")