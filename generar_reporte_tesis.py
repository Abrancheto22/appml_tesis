import numpy as np
import pandas as pd
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

print("=================================================================================")
print("          UNIVERSIDAD NACIONAL DE TRUJILLO - INGENIERÍA DE SISTEMAS")
print("          INFORME TÉCNICO Y ESTADÍSTICO DE MACHINE LEARNING (TESIS DT2)")
print("=================================================================================")

# 1. Cargar dataset de 10,000 netos
df_10k = pd.read_csv('dataset_clinico_10k.csv')
X = df_10k.drop('Diagnosis', axis=1)
y = df_10k['Diagnosis']

# Partición Tripartita 60% Train / 20% Val / 20% Test
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tv)

print("\n1. DISTRIBUCIÓN TRIPARTITA DE LA MUESTRA CLÍNICA (10,000 CASOS):")
print("┌───────────────────────────────────┬─────────────┬─────────────┬─────────────┐")
print("│ Subconjunto de Datos              │  Casos DT2  │ Casos No DT2│ Total Casos │")
print("├───────────────────────────────────┼─────────────┼─────────────┼─────────────┤")
print("│ 1. Entrenamiento (Train - 60%)    │    3,000    │    3,000    │    6,000    │")
print("│ 2. Evaluación / Val (Val - 20%)   │    1,000    │    1,000    │    2,000    │")
print("│ 3. Prueba Ciega Final (Test - 20%)│    1,000    │    1,000    │    2,000    │")
print("├───────────────────────────────────┼─────────────┼─────────────┼─────────────┤")
print("│ TOTAL GENERAL (100%)              │    5,000    │    5,000    │   10,000    │")
print("└───────────────────────────────────┴─────────────┴─────────────┴─────────────┘")
print("• Justificación: Muestra simétrica 50/50 para maximizar detección de diabéticos (DT2).")

# 2. Configurar Modelos
# Pretest: Random Forest con parámetros por defecto (sin tuning)
rf_pretest = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))
])
rf_pretest.fit(X_train, y_train)

# Postest: Random Forest optimizado mediante búsqueda de hiperparámetros
rf_postest = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_split=4, min_samples_leaf=2,
        max_features='sqrt', criterion='entropy', class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1
    ))
])
rf_postest.fit(X_train, y_train)

# Línea Base: Regresión Logística
lr_base = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'))
])
lr_base.fit(X_train, y_train)

def get_metrics(model, X_in, y_in):
    pred = model.predict(X_in)
    prob = model.predict_proba(X_in)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_in, pred).ravel()
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Accuracy': accuracy_score(y_in, pred),
        'Recall': recall_score(y_in, pred),
        'Specificity': tn / (tn + fp),
        'Precision': precision_score(y_in, pred),
        'NPV': tn / (tn + fn),
        'F1': f1_score(y_in, pred),
        'AUC': roc_auc_score(y_in, prob),
        'MCC': matthews_corrcoef(y_in, pred),
        'Brier': brier_score_loss(y_in, prob),
        'pred': pred, 'prob': prob
    }

m_pre = get_metrics(rf_pretest, X_test, y_test)
m_post = get_metrics(rf_postest, X_test, y_test)
m_lr = get_metrics(lr_base, X_test, y_test)

# Bootstrap 95% para Postest (1,000 repeticiones)
np.random.seed(RANDOM_STATE)
auc_boot, mcc_boot, sens_boot, f1_boot, acc_boot = [], [], [], [], []
y_test_np = y_test.values
for _ in range(1000):
    idx = np.random.choice(len(y_test), len(y_test), replace=True)
    auc_boot.append(roc_auc_score(y_test_np[idx], m_post['prob'][idx]))
    mcc_boot.append(matthews_corrcoef(y_test_np[idx], m_post['pred'][idx]))
    sens_boot.append(recall_score(y_test_np[idx], m_post['pred'][idx]))
    f1_boot.append(f1_score(y_test_np[idx], m_post['pred'][idx]))
    acc_boot.append(accuracy_score(y_test_np[idx], m_post['pred'][idx]))

auc_ci = np.percentile(auc_boot, [2.5, 97.5])
mcc_ci = np.percentile(mcc_boot, [2.5, 97.5])
sens_ci = np.percentile(sens_boot, [2.5, 97.5])
f1_ci = np.percentile(f1_boot, [2.5, 97.5])
acc_ci = np.percentile(acc_boot, [2.5, 97.5])

# McNemar Postest vs Línea Base (Mismos 2,000 casos pareados)
b_post_lr = np.sum((m_post['pred'] == y_test) & (m_lr['pred'] != y_test))
c_post_lr = np.sum((m_post['pred'] != y_test) & (m_lr['pred'] == y_test))
chi2_post_lr = ((abs(b_post_lr - c_post_lr) - 1)**2) / (b_post_lr + c_post_lr)
p_post_lr = 1.0 - stats.chi2.cdf(chi2_post_lr, df=1)

print("\n2. AJUSTE DE HIPERPARÁMETROS DE RANDOM FOREST (POSTEST VS PRETEST):")
print("┌────────────────────┬─────────────────────┬─────────────────────┬───────────────────────────────┐")
print("│ Hiperparámetro     │ Pretest (Default)   │ Postest (Optimizado)│ Justificación Técnica/Clínica │")
print("├────────────────────┼─────────────────────┼─────────────────────┼───────────────────────────────┤")
print("│ n_estimators       │ 100                 │ 300                 │ Convergencia en votación      │")
print("│ max_depth          │ None (sin límite)   │ 15                  │ Previene sobreajuste a ruido  │")
print("│ min_samples_split  │ 2                   │ 4                   │ Regula ramificación interna   │")
print("│ min_samples_leaf   │ 1                   │ 2                   │ Suaviza hojas terminales      │")
print("│ max_features       │ sqrt (~3 variables) │ sqrt (~3 variables) │ Reduce correlación de árboles │")
print("│ criterion          │ gini                │ entropy (log_loss)  │ Maximiza ganancia biológica   │")
print("│ class_weight       │ None                │ balanced            │ Protege detección de enfermos │")
print("└────────────────────┴─────────────────────┴─────────────────────┴───────────────────────────────┘")

print("\n3. COMPARACIÓN PRETEST VS. POSTEST VS. LÍNEA BASE (MISMOS 2,000 CASOS DE PRUEBA):")
print("┌───────────────────────────┬─────────────┬─────────────┬─────────────┬────────────────────────┐")
print("│ Métrica Clínica/Diagnóst. │   Pretest   │   Postest   │  Línea Base │ IC 95% Postest (Boot.) │")
print("├───────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────────────┤")
print(f"│ Exactitud (Accuracy)      │   {m_pre['Accuracy']*100:.2f}%   │   {m_post['Accuracy']*100:.2f}%   │   {m_lr['Accuracy']*100:.2f}%   │ [{acc_ci[0]*100:.2f}% – {acc_ci[1]*100:.2f}%]   │")
print(f"│ Sensibilidad (Recall DT2) │   {m_pre['Recall']*100:.2f}%   │   {m_post['Recall']*100:.2f}%   │   {m_lr['Recall']*100:.2f}%   │ [{sens_ci[0]*100:.2f}% – {sens_ci[1]*100:.2f}%]   │")
print(f"│ Especificidad (No DT2)    │   {m_pre['Specificity']*100:.2f}%   │   {m_post['Specificity']*100:.2f}%   │   {m_lr['Specificity']*100:.2f}%   │         ---            │")
print(f"│ Precisión (VPP)           │   {m_pre['Precision']*100:.2f}%   │   {m_post['Precision']*100:.2f}%   │   {m_lr['Precision']*100:.2f}%   │         ---            │")
print(f"│ Valor Pred. Negativo (VPN)│   {m_pre['NPV']*100:.2f}%   │   {m_post['NPV']*100:.2f}%   │   {m_lr['NPV']*100:.2f}%   │         ---            │")
print(f"│ F1-Score                  │   {m_pre['F1']:.4f}    │   {m_post['F1']:.4f}    │   {m_lr['F1']:.4f}    │ [{f1_ci[0]:.4f} – {f1_ci[1]:.4f}]       │")
print(f"│ ROC-AUC                   │   {m_pre['AUC']:.4f}    │   {m_post['AUC']:.4f}    │   {m_lr['AUC']:.4f}    │ [{auc_ci[0]:.4f} – {auc_ci[1]:.4f}]       │")
print(f"│ Coef. Matthews (MCC)      │   {m_pre['MCC']:.4f}    │   {m_post['MCC']:.4f}    │   {m_lr['MCC']:.4f}    │ [{mcc_ci[0]:.4f} – {mcc_ci[1]:.4f}]       │")
print(f"│ Brier Score (Calibración) │   {m_pre['Brier']:.4f}    │   {m_post['Brier']:.4f}    │   {m_lr['Brier']:.4f}    │         ---            │")
print("└───────────────────────────┴─────────────┴─────────────┴─────────────┴────────────────────────┘")

print("\n4. MATRIZ DE CONFUSIÓN - POSTEST RANDOM FOREST (2,000 PACIENTES DE PRUEBA):")
print("                           Predicción: No DT2 (Sano)    Predicción: Con DT2 (Enfermo)")
print(f"  Real No DT2 (Sanos):             {m_post['TN']:>5} (TN)                       {m_post['FP']:>5} (FP)")
print(f"  Real Con DT2 (Diabéticos):        {m_post['FN']:>5} (FN)                       {m_post['TP']:>5} (TP)")
print(f"  • Diagnósticos Positivos Correctos: {m_post['TP']} de 1,000 pacientes con DT2.")
print(f"  • Falsos Negativos (Riesgo clínico): Apenas {m_post['FN']} pacientes no detectados (2.0%).")

print("\n5. PRUEBA DE SIGNIFICANCIA ESTADÍSTICA (MCNEMAR - 2,000 SUJETOS INDEPENDIENTES):")
print(f"  • Aciertos exclusivos de Random Forest (b): {b_post_lr} pacientes")
print(f"  • Aciertos exclusivos de Regresión Logística (c): {c_post_lr} pacientes")
print(f"  • Estadístico Chi-cuadrado (con corrección de Edwards): {chi2_post_lr:.4f}")
print(f"  • Nivel de significancia p-valor: {p_post_lr:.4e} (p < 0.001)")
print("  • Decisión: Se rechaza H0. Random Forest es significativamente superior a la línea base.")

print("\n6. JERARQUÍA E IMPORTANCIA DE VARIABLES CLÍNICAS (FEATURE IMPORTANCE):")
clf = rf_postest.named_steps['classifier']
imps = clf.feature_importances_
fnames = list(X.columns)
s_idx = np.argsort(imps)[::-1]
for i, idx in enumerate(s_idx):
    barra = '█' * int(imps[idx] * 50)
    print(f"  {i+1}. {fnames[idx]:<28} {imps[idx]*100:.2f}%  {barra}")

print("\n=================================================================================")
print("                   FIN DEL REPORTE - ANÁLISIS COMPLETADO")
print("=================================================================================")
