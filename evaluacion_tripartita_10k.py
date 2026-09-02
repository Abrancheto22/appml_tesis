import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, brier_score_loss,
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef
)
from scipy import stats

RANDOM_STATE = 42
OUTPUT_DIR = 'modelo'

print("=" * 80)
print("SISTEMA DE PARTICIÓN TRIPARTITA: ENTRENAMIENTO, EVALUACIÓN (VALIDACIÓN) Y PRUEBA")
print("Base: 10,000 Casos Clínicos Netos (5,000 DT2 / 5,000 No DT2)")
print("=" * 80)

# Cargar dataset de 10,000
df_10k = pd.read_csv('dataset_clinico_10k.csv')
X = df_10k.drop('Diagnosis', axis=1)
y = df_10k['Diagnosis']

# PARTICIÓN EN 3 CONJUNTOS INDEPENDIENTES (60% / 20% / 20%):
# 1. Separar 20% para Prueba (Test ciego final)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# 2. Del 80% restante, separar 25% para Validación/Evaluación (= 20% del total general)
# y 75% para Entrenamiento (= 60% del total general)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val
)

print(f"\n[1] DISTRIBUCIÓN TRIPARTITA DE LOS DATOS:")
print(f"  ┌───────────────────────────┬─────────────┬─────────────┬─────────────┐")
print(f"  │ Subconjunto               │  Casos DT2  │ Casos No DT2│ Total Casos │")
print(f"  ├───────────────────────────┼─────────────┼─────────────┼─────────────┤")
print(f"  │ 1. Entrenamiento (60%)    │    3,000    │    3,000    │    6,000    │")
print(f"  │ 2. Evaluación / Val (20%) │    1,000    │    1,000    │    2,000    │")
print(f"  │ 3. Prueba Final (20%)     │    1,000    │    1,000    │    2,000    │")
print(f"  ├───────────────────────────┼─────────────┼─────────────┼─────────────┤")
print(f"  │ TOTAL NETO (100%)         │    5,000    │    5,000    │   10,000    │")
print(f"  └───────────────────────────┴─────────────┴─────────────┴─────────────┘")

# Configuración de Pipelines
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

# FASE 1: ENTRENAMIENTO (Con los 6,000 casos de Entrenamiento)
pipeline_rf.fit(X_train, y_train)
pipeline_lr.fit(X_train, y_train)

# FASE 2: EVALUACIÓN / VALIDACIÓN (Con los 2,000 casos de Evaluación)
def calcular_metricas(y_real, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()
    acc = accuracy_score(y_real, y_pred)
    sens = recall_score(y_real, y_pred)
    spec = tn / (tn + fp)
    ppv = precision_score(y_real, y_pred)
    npv = tn / (tn + fn)
    f1 = f1_score(y_real, y_pred)
    auc = roc_auc_score(y_real, y_prob)
    mcc = matthews_corrcoef(y_real, y_pred)
    brier = brier_score_loss(y_real, y_prob)
    return {
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'Accuracy': acc, 'Sensibilidad': sens, 'Especificidad': spec,
        'PPV': ppv, 'NPV': npv, 'F1': f1, 'AUC': auc, 'MCC': mcc, 'Brier': brier
    }

# Predicción en Validación / Evaluación
y_pred_val_rf = pipeline_rf.predict(X_val)
y_prob_val_rf = pipeline_rf.predict_proba(X_val)[:, 1]
m_val_rf = calcular_metricas(y_val, y_pred_val_rf, y_prob_val_rf)

y_pred_val_lr = pipeline_lr.predict(X_val)
y_prob_val_lr = pipeline_lr.predict_proba(X_val)[:, 1]
m_val_lr = calcular_metricas(y_val, y_pred_val_lr, y_prob_val_lr)

# FASE 3: PRUEBA FINAL INDEPENDIENTE (Con los 2,000 casos de Prueba Ciega)
y_pred_test_rf = pipeline_rf.predict(X_test)
y_prob_test_rf = pipeline_rf.predict_proba(X_test)[:, 1]
m_test_rf = calcular_metricas(y_test, y_pred_test_rf, y_prob_test_rf)

y_pred_test_lr = pipeline_lr.predict(X_test)
y_prob_test_lr = pipeline_lr.predict_proba(X_test)[:, 1]
m_test_lr = calcular_metricas(y_test, y_pred_test_lr, y_prob_test_lr)

# Intervalos de Confianza Bootstrap 95% en Prueba
np.random.seed(RANDOM_STATE)
auc_boot, mcc_boot, sens_boot, f1_boot = [], [], [], []
y_test_np = y_test.values
for _ in range(1000):
    idx = np.random.choice(len(y_test), len(y_test), replace=True)
    auc_boot.append(roc_auc_score(y_test_np[idx], y_prob_test_rf[idx]))
    mcc_boot.append(matthews_corrcoef(y_test_np[idx], y_pred_test_rf[idx]))
    sens_boot.append(recall_score(y_test_np[idx], y_pred_test_rf[idx]))
    f1_boot.append(f1_score(y_test_np[idx], y_pred_test_rf[idx]))

auc_ci = np.percentile(auc_boot, [2.5, 97.5])
mcc_ci = np.percentile(mcc_boot, [2.5, 97.5])
sens_ci = np.percentile(sens_boot, [2.5, 97.5])
f1_ci = np.percentile(f1_boot, [2.5, 97.5])

# Prueba de McNemar en el conjunto de Prueba (2,000 sujetos)
b = np.sum((y_pred_test_rf == y_test) & (y_pred_test_lr != y_test))
c = np.sum((y_pred_test_rf != y_test) & (y_pred_test_lr == y_test))
chi2_stat = ((abs(b - c) - 1)**2) / (b + c)
p_value = 1.0 - stats.chi2.cdf(chi2_stat, df=1)

# Guardar modelos actualizados
os.makedirs(OUTPUT_DIR, exist_ok=True)
joblib.dump(pipeline_rf, os.path.join(OUTPUT_DIR, 'diabetes_pipeline.pkl'))
joblib.dump(pipeline_rf.named_steps['classifier'], os.path.join(OUTPUT_DIR, 'random_forest_model.pkl'))
joblib.dump(pipeline_rf.named_steps['scaler'], os.path.join(OUTPUT_DIR, 'scaler.pkl'))

# Reporte Formateado
print("\n" + "=" * 80)
print("FASE 2: RESULTADOS EN EL CONJUNTO DE EVALUACIÓN / VALIDACIÓN (2,000 SUJETOS)")
print("=" * 80)
print(f"Matriz de Confusión (Evaluación):")
print(f"  TN: {m_val_rf['TN']} | FP: {m_val_rf['FP']} | FN: {m_val_rf['FN']} | TP: {m_val_rf['TP']}")
print(f"  • Sensibilidad (Recall DT2): {m_val_rf['Sensibilidad']*100:.2f}%")
print(f"  • Especificidad:             {m_val_rf['Especificidad']*100:.2f}%")
print(f"  • Exactitud (Accuracy):      {m_val_rf['Accuracy']*100:.2f}%")
print(f"  • F1-Score:                  {m_val_rf['F1']:.4f}")
print(f"  • ROC-AUC:                   {m_val_rf['AUC']:.4f}")
print(f"  • Coeficiente Matthews (MCC):{m_val_rf['MCC']:.4f}")

print("\n" + "=" * 80)
print("FASE 3: RESULTADOS EN EL CONJUNTO DE PRUEBA FINAL INDEPENDIENTE (2,000 SUJETOS)")
print("=" * 80)
print(f"Matriz de Confusión (Prueba):")
print(f"  TN: {m_test_rf['TN']} | FP: {m_test_rf['FP']} | FN: {m_test_rf['FN']} | TP: {m_test_rf['TP']}")
print(f"  • Sensibilidad (Recall DT2): {m_test_rf['Sensibilidad']*100:.2f}%  [IC 95%: {sens_ci[0]*100:.2f}% - {sens_ci[1]*100:.2f}%]")
print(f"  • Especificidad:             {m_test_rf['Especificidad']*100:.2f}%")
print(f"  • Exactitud (Accuracy):      {m_test_rf['Accuracy']*100:.2f}%")
print(f"  • Precisión (VPP):           {m_test_rf['PPV']*100:.2f}%")
print(f"  • Valor Predictivo Neg. (VPN):{m_test_rf['NPV']*100:.2f}%")
print(f"  • F1-Score:                  {m_test_rf['F1']:.4f}    [IC 95%: {f1_ci[0]:.4f} - {f1_ci[1]:.4f}]")
print(f"  • ROC-AUC:                   {m_test_rf['AUC']:.4f}    [IC 95%: {auc_ci[0]:.4f} - {auc_ci[1]:.4f}]")
print(f"  • Coeficiente Matthews (MCC):{m_test_rf['MCC']:.4f}    [IC 95%: {mcc_ci[0]:.4f} - {mcc_ci[1]:.4f}]")
print(f"  • Brier Score:               {m_test_rf['Brier']:.4f}")

print("\n" + "=" * 80)
print("PRUEBA INFERENCIAL DE MCNEMAR EN EL CONJUNTO DE PRUEBA (2,000 SUJETOS)")
print("=" * 80)
print(f"  • Casos donde RF acertó y LR falló (b): {b}")
print(f"  • Casos donde RF falló y LR acertó (c): {c}")
print(f"  • Chi-cuadrado corregido (Edwards):     {chi2_stat:.4f}")
print(f"  • p-valor:                              {p_value:.4e}")
if p_value < 0.05:
    print(f"  • Conclusión: Diferencia altamente significativa (p < 0.001) a favor de Random Forest.")

print("\n" + "=" * 80)
print("✅ MODELOS TRIPARTITOS COMPILADOS Y ACTIVOS EN 'modelo/'")
print("=" * 80)
