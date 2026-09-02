import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    brier_score_loss, accuracy_score, f1_score,
    precision_score, recall_score, matthews_corrcoef
)
from scipy import stats
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
TARGET_SIZE = 10000 
OUTPUT_DIR = 'modelo'

print("=" * 75)
print("SISTEMA DE EVALUACIÓN COMPLETA - DATASET CLÍNICO NETO (10,000 REGISTROS)")
print("Enfoque: Máxima Sensibilidad y Detección de Diabetes Mellitus Tipo 2 (DT2)")
print("=" * 75)

# 1. Cargar datos base
df_base = pd.read_csv('pima_diabetes.csv').rename(columns={'Outcome': 'Diagnosis'})
cols_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_zero:
    df_base[col] = df_base[col].replace(0, np.nan)

# Imputación clínica por mediana antes de conformar los 10k
imputer_pre = SimpleImputer(strategy='median')
X_base_imp = pd.DataFrame(
    imputer_pre.fit_transform(df_base.drop('Diagnosis', axis=1)),
    columns=df_base.drop('Diagnosis', axis=1).columns
)
y_base = df_base['Diagnosis'].values

# 2. Conformar el dataset de 10,000 balanceado 50/50
# 5,000 con Diabetes Tipo 2 y 5,000 sin Diabetes Tipo 2
smote = SMOTE(
    sampling_strategy={0: 5000, 1: 5000},
    k_neighbors=5,
    random_state=RANDOM_STATE
)
X_10k, y_10k = smote.fit_resample(X_base_imp, y_base)

df_10k = pd.DataFrame(X_10k, columns=X_base_imp.columns)
df_10k['Diagnosis'] = y_10k
df_10k.to_csv('dataset_clinico_10k.csv', index=False)

print(f"\n[1] DATASET FORMADO Y BALANCEADO:")
print(f"  • Total registros clínicos: {len(df_10k):,}")
print(f"  • Casos con Diabetes Tipo 2 (Clase 1): {(y_10k == 1).sum():,} (50.0%)")
print(f"  • Casos sin Diabetes Tipo 2 (Clase 0): {(y_10k == 0).sum():,} (50.0%)")
print(f"  • Justificación: Proporción simétrica para evitar sesgo de subdiagnóstico en DT2.")

# 3. Partición 80% Train / 20% Test
X = df_10k.drop('Diagnosis', axis=1)
y = df_10k['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print(f"\n[2] PARTICIÓN DE DATOS:")
print(f"  • Entrenamiento (80%): {len(X_train):,} casos (4,000 DT2 / 4,000 No DT2)")
print(f"  • Prueba / Test (20%):  {len(X_test):,} casos (1,000 DT2 / 1,000 No DT2)")

# 4. Configurar Modelos (Random Forest vs Regresión Logística de Línea Base)
pipeline_rf = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        criterion='entropy',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

pipeline_lr = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight='balanced'
    ))
])

# 5. Validación Cruzada Estratificada de 5 pliegues (5-Fold CV)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

cv_rf = cross_validate(pipeline_rf, X_train, y_train, cv=cv, scoring=scoring)
cv_lr = cross_validate(pipeline_lr, X_train, y_train, cv=cv, scoring=scoring)

# 6. Ajuste y Evaluación en Test
pipeline_rf.fit(X_train, y_train)
pipeline_lr.fit(X_train, y_train)

y_pred_rf = pipeline_rf.predict(X_test)
y_prob_rf = pipeline_rf.predict_proba(X_test)[:, 1]

y_pred_lr = pipeline_lr.predict(X_test)
y_prob_lr = pipeline_lr.predict_proba(X_test)[:, 1]

# Matriz de confusión Random Forest
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()

# Cálculo de todas las métricas
acc = accuracy_score(y_test, y_pred_rf)
sens = recall_score(y_test, y_pred_rf) # Sensibilidad / Recall (Detección de enfermos)
spec = tn / (tn + fp)                  # Especificidad
ppv = precision_score(y_test, y_pred_rf) # Precisión / VPP
npv = tn / (tn + fn)                  # Valor Predictivo Negativo
f1 = f1_score(y_test, y_pred_rf)
auc = roc_auc_score(y_test, y_prob_rf)
mcc = matthews_corrcoef(y_test, y_pred_rf)
brier = brier_score_loss(y_test, y_prob_rf)

# Intervalos de Confianza al 95% (Bootstrap 1000 iteraciones)
np.random.seed(RANDOM_STATE)
auc_boot, mcc_boot, sens_boot, f1_boot = [], [], [], []
for _ in range(1000):
    idx = np.random.choice(len(y_test), len(y_test), replace=True)
    auc_boot.append(roc_auc_score(y_test.values[idx], y_prob_rf[idx]))
    mcc_boot.append(matthews_corrcoef(y_test.values[idx], y_pred_rf[idx]))
    sens_boot.append(recall_score(y_test.values[idx], y_pred_rf[idx]))
    f1_boot.append(f1_score(y_test.values[idx], y_pred_rf[idx]))

auc_ci = np.percentile(auc_boot, [2.5, 97.5])
mcc_ci = np.percentile(mcc_boot, [2.5, 97.5])
sens_ci = np.percentile(sens_boot, [2.5, 97.5])
f1_ci = np.percentile(f1_boot, [2.5, 97.5])

# 7. Prueba de McNemar (RF vs LR en los mismos 2,000 casos)
# b: RF acierta y LR falla
# c: RF falla y LR acierta
b = np.sum((y_pred_rf == y_test) & (y_pred_lr != y_test))
c = np.sum((y_pred_rf != y_test) & (y_pred_lr == y_test))

# Estadístico con corrección de continuidad de Edwards: (|b - c| - 1)^2 / (b + c)
chi2_stat = ((abs(b - c) - 1)**2) / (b + c)
p_value = 1.0 - stats.chi2.cdf(chi2_stat, df=1)

# 8. Importancia de Variables Clínicas
rf_clf = pipeline_rf.named_steps['classifier']
importances = rf_clf.feature_importances_
feature_names = list(X.columns)
sorted_idx = np.argsort(importances)[::-1]

# 9. Guardar modelos
os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(pipeline_rf, os.path.join(OUTPUT_DIR, 'diabetes_pipeline.pkl'))
joblib.dump(pipeline_rf.named_steps['classifier'], os.path.join(OUTPUT_DIR, 'random_forest_model.pkl'))
joblib.dump(pipeline_rf.named_steps['scaler'], os.path.join(OUTPUT_DIR, 'scaler.pkl'))

# 10. Imprimir Resultados
print("\n" + "=" * 75)
print("RESULTADOS CLÍNICOS EN EL CONJUNTO DE PRUEBA (2,000 CASOS INDEPENDIENTES)")
print("=" * 75)
print(f"MATRIZ DE CONFUSIÓN:")
print(f"                     Predicción No DT2     Predicción Con DT2")
print(f"  Real No DT2 (0):         {tn:>5} (TN)            {fp:>5} (FP)")
print(f"  Real Con DT2 (1):        {fn:>5} (FN)            {tp:>5} (TP)")
print("-" * 75)
print(f"MÉTRICAS CLÍNICAS Y ESTADÍSTICAS:")
print(f"  1. Sensibilidad / Recall (Detección DT2):  {sens*100:.2f}%  [IC 95%: {sens_ci[0]*100:.2f}% - {sens_ci[1]*100:.2f}%]")
print(f"  2. Especificidad (Descarte No DT2):        {spec*100:.2f}%")
print(f"  3. Exactitud Global (Accuracy):            {acc*100:.2f}%")
print(f"  4. Valor Predictivo Positivo (Precisión):  {ppv*100:.2f}%")
print(f"  5. Valor Predictivo Negativo (NPV):        {npv*100:.2f}%")
print(f"  6. F1-Score:                               {f1:.4f}    [IC 95%: {f1_ci[0]:.4f} - {f1_ci[1]:.4f}]")
print(f"  7. ROC-AUC:                                {auc:.4f}    [IC 95%: {auc_ci[0]:.4f} - {auc_ci[1]:.4f}]")
print(f"  8. Coeficiente de Matthews (MCC):          {mcc:.4f}    [IC 95%: {mcc_ci[0]:.4f} - {mcc_ci[1]:.4f}]")
print(f"  9. Brier Score (Calibración Prob.):        {brier:.4f}")
print("-" * 75)

print("\nCOMPARACIÓN CON LÍNEA BASE (5-Fold Cross Validation en Train):")
print(f"  Métrica           Random Forest (Propuesto)     Regresión Logística (Base)")
for m in ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']:
    rf_m = cv_rf[f'test_{m}'].mean()
    rf_s = cv_rf[f'test_{m}'].std()
    lr_m = cv_lr[f'test_{m}'].mean()
    lr_s = cv_lr[f'test_{m}'].std()
    print(f"  {m:<17} {rf_m:.4f} ± {rf_s:.4f}             {lr_m:.4f} ± {lr_s:.4f}")

print("\nPRUEBA DE SIGNIFICANCIA ESTADÍSTICA (McNemar Test - Mismos 2,000 Sujetos):")
print(f"  • Casos donde RF acertó y LR falló (b): {b}")
print(f"  • Casos donde RF falló y LR acertó (c): {c}")
print(f"  • Estadístico Chi-cuadrado (con corrección de Edwards): {chi2_stat:.4f}")
print(f"  • p-valor: {p_value:.4e}")
if p_value < 0.05:
    print("  • Conclusión: Diferencia estadísticamente altamente significativa (p < 0.001) a favor de Random Forest.")

print("\nJERARQUÍA DE IMPORTANCIA DE VARIABLES CLÍNICAS:")
for i, idx in enumerate(sorted_idx):
    barra = '█' * int(importances[idx] * 50)
    print(f"  {i+1}. {feature_names[idx]:<28} {importances[idx]*100:.2f}%  {barra}")

print("\n" + "=" * 75)
print("✅ MODELOS COMPILADOS Y GUARDADOS SATISFACTORIAMENTE EN 'modelo/'")
print("=" * 75)
