#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
    roc_auc_score, brier_score_loss, f1_score, precision_score, recall_score, accuracy_score)
from scipy.stats import randint

RANDOM_STATE = 42
TEST_SIZE    = 0.2
VAL_SIZE     = 0.25
CV_FOLDS     = 5
N_ITER       = 100
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pima_diabetes.csv')
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelo')

print("=" * 70)
print("ENTRENAMIENTO — PIMA INDIANS DIABETES DATASET (UCI / Smith et al. 1988)")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)
df = df.rename(columns={'Outcome': 'Diagnosis'})

print(f"\n[1/7] Dataset cargado: {len(df)} registros")
print(f"  Sin diabetes (0): {(df['Diagnosis']==0).sum()} ({(df['Diagnosis']==0).mean()*100:.1f}%)")
print(f"  Con diabetes (1): {(df['Diagnosis']==1).sum()} ({(df['Diagnosis']==1).mean()*100:.1f}%)")

cols_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for col in cols_zero:
    n = (df[col] == 0).sum()
    if n > 0:
        df[col] = df[col].replace(0, np.nan)

print("\n  Correlaciones con Diagnosis:")
for col, val in df.corr()['Diagnosis'].drop('Diagnosis').items():
    print(f"    {col:<30} r={val:+.4f}")

X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']
feature_names = list(X.columns)

X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_tv)
print(f"\n[2/7] División: Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

pipeline_rf = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
])
pipeline_lr = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'))
])

param_dist = {
    'classifier__n_estimators': randint(100, 500),
    'classifier__max_depth': [5, 10, 15, 20, 25, None],
    'classifier__min_samples_split': randint(2, 20),
    'classifier__min_samples_leaf': randint(1, 10),
    'classifier__max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
    'classifier__criterion': ['gini', 'entropy', 'log_loss'],
    'classifier__class_weight': [None, 'balanced', 'balanced_subsample'],
    'classifier__bootstrap': [True, False],
}

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
print(f"\n[4/7] RandomizedSearchCV ({N_ITER} iteraciones)...")
search = RandomizedSearchCV(pipeline_rf, param_dist, n_iter=N_ITER, cv=cv,
    scoring='roc_auc', random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
search.fit(X_train, y_train)

print(f"\n  Mejores hiperparámetros:")
for p, v in search.best_params_.items():
    print(f"    {p.replace('classifier__','')}: {v}")
print(f"  Mejor ROC-AUC (CV): {search.best_score_:.4f}")

scoring = {'accuracy':'accuracy','precision':'precision','recall':'recall','f1':'f1','roc_auc':'roc_auc'}
cv_rf = cross_validate(search.best_estimator_, X_train, y_train, cv=cv, scoring=scoring)
cv_lr = cross_validate(pipeline_lr, X_train, y_train, cv=cv, scoring=scoring)

print(f"\n[5/7] Validación cruzada:")
print(f"  {'Métrica':<12} {'Random Forest':>20} {'Reg. Logística':>20}")
print(f"  {'─'*55}")
for m in ['accuracy','precision','recall','f1','roc_auc']:
    rf_m = cv_rf[f'test_{m}'].mean(); rf_s = cv_rf[f'test_{m}'].std()
    lr_m = cv_lr[f'test_{m}'].mean(); lr_s = cv_lr[f'test_{m}'].std()
    print(f"  {m:<12} {rf_m:.4f} ± {rf_s:.4f}  {lr_m:.4f} ± {lr_s:.4f}")

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:,1]
tn,fp,fn,tp = confusion_matrix(y_test, y_pred).ravel()

accuracy    = (tp+tn)/(tp+tn+fp+fn)
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
ppv         = tp/(tp+fp)
npv         = tn/(tn+fn)
f1          = 2*(ppv*sensitivity)/(ppv+sensitivity)
auc         = roc_auc_score(y_test, y_prob)
brier       = brier_score_loss(y_test, y_prob)

np.random.seed(RANDOM_STATE)
auc_b, f1_b = [], []
for _ in range(1000):
    idx = np.random.choice(len(y_test), len(y_test), replace=True)
    auc_b.append(roc_auc_score(y_test.values[idx], y_prob[idx]))
    f1_b.append(f1_score(y_test.values[idx], y_pred[idx]))
auc_lo,auc_hi = np.percentile(auc_b,[2.5,97.5])
f1_lo,f1_hi   = np.percentile(f1_b, [2.5,97.5])

print(f"\n[6/7] MÉTRICAS EN TEST:")
print(f"  {'─'*50}")
print(f"  Accuracy (Exactitud):  {accuracy:.4f}")
print(f"  Sensibilidad (Recall): {sensitivity:.4f}")
print(f"  Especificidad:         {specificity:.4f}")
print(f"  Precisión (PPV):       {ppv:.4f}")
print(f"  NPV:                   {npv:.4f}")
print(f"  F1-Score:              {f1:.4f}  (IC 95%: {f1_lo:.4f} – {f1_hi:.4f})")
print(f"  ROC-AUC:               {auc:.4f}  (IC 95%: {auc_lo:.4f} – {auc_hi:.4f})")
print(f"  Brier Score:           {brier:.4f}")
print(f"  {'─'*50}")
print(f"\n  MATRIZ DE CONFUSIÓN:")
print(f"  {'':>14} {'Pred: Neg':>12} {'Pred: Pos':>12}")
print(f"  {'Real: Neg':>14} {tn:>12} {fp:>12}")
print(f"  {'Real: Pos':>14} {fn:>12} {tp:>12}")
print(f"\n  REPORTE:")
print(classification_report(y_test, y_pred, target_names=['Sin diabetes','Diabetes']))

rf_clf = best_model.named_steps['classifier']
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]
print(f"\n[7/7] Importancia de variables:")
for i, idx in enumerate(indices):
    barra = '█' * int(importances[idx]*60)
    print(f"  {i+1}. {feature_names[idx]:<30} {importances[idx]:.4f}  {barra}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(best_model, os.path.join(OUTPUT_DIR, 'diabetes_pipeline.pkl'))
joblib.dump(best_model.named_steps['classifier'], os.path.join(OUTPUT_DIR, 'random_forest_model.pkl'))
joblib.dump(best_model.named_steps['scaler'], os.path.join(OUTPUT_DIR, 'scaler.pkl'))

bp = search.best_params_
rf_auc = cv_rf['test_roc_auc'].mean()
lr_auc = cv_lr['test_roc_auc'].mean()
print("\n" + "="*70)
print("RESUMEN PARA INFORME DE TESIS")
print("="*70)
print(f"""
FUENTE: PIMA Indians Diabetes Dataset — Smith et al. (1988), UCI.
  Registros: {len(df)} | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}
  Sin diabetes: {(df['Diagnosis']==0).sum()} ({(df['Diagnosis']==0).mean()*100:.1f}%)
  Con diabetes: {(df['Diagnosis']==1).sum()} ({(df['Diagnosis']==1).mean()*100:.1f}%)

HIPERPARÁMETROS (RandomizedSearchCV, {N_ITER} iter., {CV_FOLDS}-Fold Stratified CV):
  n_estimators:      {bp.get('classifier__n_estimators')}
  max_depth:         {bp.get('classifier__max_depth')}
  min_samples_split: {bp.get('classifier__min_samples_split')}
  min_samples_leaf:  {bp.get('classifier__min_samples_leaf')}
  max_features:      {bp.get('classifier__max_features')}
  criterion:         {bp.get('classifier__criterion')}
  class_weight:      {bp.get('classifier__class_weight')}
  bootstrap:         {bp.get('classifier__bootstrap')}

RESULTADOS EN TEST:
  Accuracy:      {accuracy:.4f}
  Sensibilidad:  {sensitivity:.4f}
  Especificidad: {specificity:.4f}
  PPV:           {ppv:.4f}
  NPV:           {npv:.4f}
  F1-Score:      {f1:.4f}  IC95%: [{f1_lo:.4f} – {f1_hi:.4f}]
  ROC-AUC:       {auc:.4f}  IC95%: [{auc_lo:.4f} – {auc_hi:.4f}]
  Brier Score:   {brier:.4f}

COMPARACIÓN LÍNEA BASE (5-Fold CV):
  Random Forest ROC-AUC:    {rf_auc:.4f}
  Reg. Logística ROC-AUC:   {lr_auc:.4f}
  Ganador: {'Random Forest' if rf_auc >= lr_auc else 'Reg. Logística'}

LIBRERÍAS: scikit-learn {__import__('sklearn').__version__}, numpy {np.__version__}, pandas {pd.__version__}
""")
print("✅ Entrenamiento PIMA completado. Modelos guardados en:", OUTPUT_DIR)
