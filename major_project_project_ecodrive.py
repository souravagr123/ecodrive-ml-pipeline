import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

np.random.seed(42)

# --- 1. DATA LOADING & CLEANING ---
df = pd.read_csv('ecodrive_ev_dataset.csv')

numerical_cols = ['Battery_Capacity_kWh', 'Charging_Speed_kW', 'Max_Range_km', 'Vehicle_Age_Years', 'Odometer_km', 'Resale_Price_USD']
categorical_cols = ['Brand', 'Battery_Chemistry', 'Region']

# Impute Missing Values
df[numerical_cols] = SimpleImputer(strategy='median').fit_transform(df[numerical_cols])
df[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[categorical_cols])

# IQR Outlier Filter
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    return data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]

df_clean = remove_outliers_iqr(df, 'Odometer_km')
df_clean = remove_outliers_iqr(df_clean, 'Resale_Price_USD')

# --- 2. FEATURE ENGINEERING & PREPROCESSING ---
df_clean['Energy_Efficiency_Index'] = (
    df_clean['Max_Range_km'] / df_clean['Battery_Capacity_kWh']
) * (1 - (df_clean['Vehicle_Age_Years'] * 0.02))

X = df_clean.drop(columns=['High_Demand', 'Resale_Price_USD'])
y_class = df_clean['High_Demand']
y_reg = df_clean['Resale_Price_USD']

numerical_features = ['Battery_Capacity_kWh', 'Charging_Speed_kW', 'Max_Range_km', 'Vehicle_Age_Years', 'Odometer_km', 'Energy_Efficiency_Index']
categorical_features = ['Brand', 'Battery_Chemistry', 'Region']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)
X_processed = preprocessor.fit_transform(X)

# --- 3. FEATURE SELECTION (RFE) ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rfe_clf = RFE(estimator=LogisticRegression(max_iter=1000, random_state=42), n_features_to_select=8, step=1)
X_selected_class = rfe_clf.fit_transform(X_processed, y_class)

rfe_reg = RFE(estimator=LinearRegression(), n_features_to_select=8, step=1)
X_selected_reg = rfe_reg.fit_transform(X_processed, y_reg)

# --- 4. HYPERPARAMETER TUNING & ENSEMBLE TRAINING ---
param_distributions = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

clf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=10, cv=skf, scoring='accuracy', random_state=42, n_jobs=-1
)
clf_search.fit(X_selected_class, y_class)
best_rf_clf = clf_search.best_estimator_

reg_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_distributions,
    n_iter=10, cv=kf, scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1
)
reg_search.fit(X_selected_reg, y_reg)
best_rf_reg = reg_search.best_estimator_

# --- 5. EXPORT MODEL ARTIFACTS (.pkl) ---
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(rfe_clf, 'rfe_clf.pkl')
joblib.dump(rfe_reg, 'rfe_reg.pkl')
joblib.dump(best_rf_clf, 'best_rf_clf.pkl')
joblib.dump(best_rf_reg, 'best_rf_reg.pkl')

print("Pipeline execution complete. All 5 model artifacts (.pkl) exported successfully.")