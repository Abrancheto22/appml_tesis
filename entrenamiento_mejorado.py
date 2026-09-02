#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
SCRIPT DE ENTRENAMIENTO MEJORADO — Tesis: Predicción Diabetes Tipo 2
Centro de Salud Casa Grande
=============================================================================
Este script reemplaza el pipeline de entrenamiento original con las
correcciones solicitadas por el informe de revisión:

  1. Pipeline de scikit-learn (sin fuga de información)
  2. Justificación de Random Forest con comparación contra línea base
  3. Búsqueda de hiperparámetros con RandomizedSearchCV
  4. Validación cruzada estratificada
  5. Métricas completas con intervalos de confianza
  6. Serialización del pipeline completo

Autor: Ordoñez Reyes A.B. & Quispe Sánchez E.S.
Fecha: Agosto 2026
=============================================================================
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV,
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    brier_score_loss, accuracy_score, precision_score, recall_score,
    f1_score, precision_recall_curve, roc_curve
)
from scipy.stats import randint, uniform
import os

# =========================================================================
# CONFIGURACIÓN
# =========================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25  # 25% del 80% restante = 20% del total
CV_FOLDS = 5
N_ITER_SEARCH = 100  # Iteraciones de RandomizedSearchCV
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'Diabetes_prediction.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'modelo')

print("=" * 70)
print("ENTRENAMIENTO MEJORADO — PREDICCIÓN DIABETES TIPO 2")
print("=" * 70)

# =========================================================================
# 1. CARGA Y ANÁLISIS EXPLORATORIO
# =========================================================================
print("\n[1/7] Cargando dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"  Registros totales: {len(df)}")
print(f"  Variables: {list(df.columns)}")

# Distribución de clases
print(f"\n  Distribución de clases:")
for cls, count in df['Diagnosis'].value_counts().items():
    pct = count / len(df) * 100
    label = "Sin diabetes" if cls == 0 else "Diabetes"
    print(f"    Clase {cls} ({label}): {count} ({pct:.1f}%)")

# Verificar valores faltantes y negativos
print(f"\n  Valores faltantes: {df.isnull().sum().sum()}")
print(f"  Registros con Insulina negativa: {(df['Insulin'] < 0).sum()}")
print(f"  Registros con Edad negativa: {(df['Age'] < 0).sum()}")

# Separar features y target
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

feature_names = list(X.columns)
print(f"\n  Features: {feature_names}")
print(f"  Shape X: {X.shape}, Shape y: {y.shape}")

# =========================================================================
# 2. DIVISIÓN DE DATOS (SIN FUGA DE INFORMACIÓN)
# =========================================================================
print("\n[2/7] Dividiendo datos (estratificado, sin fuga)...")

# Primera división: 80% train+val, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Segunda división: 60% train, 20% val (del 80% original)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=VAL_SIZE,
    random_state=RANDOM_STATE, stratify=y_train_val
)

print(f"  Train: {len(X_train)} registros ({len(X_train)/len(X)*100:.0f}%)")
print(f"  Validación: {len(X_val)} registros ({len(X_val)/len(X)*100:.0f}%)")
print(f"  Test: {len(X_test)} registros ({len(X_test)/len(X)*100:.0f}%)")

# Verificar distribución de clases en cada conjunto
print(f"\n  Distribución en Train:")
for cls in [0, 1]:
    count = (y_train == cls).sum()
    print(f"    Clase {cls}: {count} ({count/len(y_train)*100:.1f}%)")

# =========================================================================
# 3. JUSTIFICACIÓN DE RANDOM FOREST
# =========================================================================
print("\n" + "=" * 70)
print("JUSTIFICACIÓN DEL ALGORITMO SELECCIONADO: RANDOM FOREST")
print("=" * 70)
print("""
Random Forest fue seleccionado como algoritmo principal por las siguientes
razones, sustentadas en la literatura y las características del problema:

1. ROBUSTEZ ANTE SOBREAJUSTE: Al ser un ensemble de múltiples árboles de
   decisión, reduce la varianza y el riesgo de sobreajuste comparado con un
   único árbol de decisión. Cada árboles se entrena con una muestra bootstrap
   diferente y un subconjunto aleatorio de variables.

2. MANEJO DE DATOS CLÍNICOS: Los datos de salud frecuentemente contienen
   valores atípicos, relaciones no lineales e interacciones complejas entre
   variables. Random Forest captura estas relaciones sin requerir supuestos
   de linealidad ni normalidad.

3. VERSATILIDAD: Funciona tanto para clasificación como regresión, maneja
   variables numéricas y categóricas, y es robusto ante variables
   irrelevantes (las descarta automáticamente).

4. INTERPRETABILIDAD PARCIAL: A través de la importancia de variables
   (feature_importances_) permite identificar qué factores clínicos contribuyen
   más a la predicción, lo cual es valioso en un contexto médico.

5. EVIDENCIA EN LITERATURA: Múltiples estudios previos han demostrado el
   rendimiento de Random Forest en predicción de diabetes:
   - Berrios Zúñiga (2024): F1 Score mejorado en 12% con PIMA
   - García-Ríos et al. (2023): CRISP-DM con Random Forest para DM2
   - Posadas Ruiz (2022): Random Forest + SMOTE para predicción DM2
""")

# =========================================================================
# 4. DEFINICIÓN DE PIPELINES Y LÍNEA BASE
# =========================================================================
print("[3/7] Definiendo pipelines...")

# Pipeline para Random Forest
pipeline_rf = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=RANDOM_STATE))
])

# Línea base: Regresión Logística (requerida por informe de revisión)
pipeline_lr = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'
    ))
])

# =========================================================================
# 5. BÚSQUEDA DE HIPERPARÁMETROS (RandomizedSearchCV)
# =========================================================================
print("\n[4/7] Buscando mejores hiperparámetros (RandomizedSearchCV)...")

# Espacio de búsqueda para Random Forest
param_distributions_rf = {
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

search_rf = RandomizedSearchCV(
    pipeline_rf,
    param_distributions_rf,
    n_iter=N_ITER_SEARCH,
    cv=cv,
    scoring='f1',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True,
    verbose=0,
    return_train_score=True
)

search_rf.fit(X_train, y_train)

print(f"\n  Mejores hiperparámetros encontrados:")
for param, value in search_rf.best_params_.items():
    clean_name = param.replace('classifier__', '')
    print(f"    {clean_name}: {value}")

print(f"\n  Mejor F1-Score (CV): {search_rf.best_score_:.4f}")

# Guardar los 3 hiperparámetros principales con justificación
best_params = search_rf.best_params_
print("\n  HIPERPARÁMETROS SELECCIONADOS Y JUSTIFICACIÓN:")
print(f"  {'─' * 60}")

justificaciones = {
    'n_estimators': (
        "Número de árboles en el bosque. Más árboles = mayor estabilidad "
        "y precisión, pero mayor costo computacional. Típico: 100-500."
    ),
    'max_depth': (
        "Profundidad máxima de cada árbo. Controla directamente el "
        "sobreajuste. None = sin límite (cada árbol crece hasta hojas puras). "
        "Limitar a 5-20 puede mejorar generalización."
    ),
    'min_samples_split': (
        "Mínimo de muestras requeridas para dividir un nodo interno. "
        "Valores más altos (5-20) previenen sobreajuste en datasets pequeños "
        "y hacen que los árboles sean más conservadores."
    ),
    'min_samples_leaf': (
        "Mínimo de muestras en una hoja terminal. Valores 2-4 suavizan "
        "el modelo, reducen varianza y previenen que el modelo memorice "
        "ruido en los datos de entrenamiento."
    ),
    'max_features': (
        "Número de variables consideradas en cada división. 'sqrt' = √8 ≈ 3 "
        "para este dataset. Reducir este valor增加了 la diversidad entre árboles "
        "y reduce la correlación entre ellos."
    ),
    'criterion': (
        "Función de medición de impureza. 'gini' (default) es más rápido; "
        "'entropy' o 'log_loss' pueden mejorar marginalmente en algunos casos."
    ),
    'class_weight': (
        "Peso de las clases. 'balanced' ajusta automáticamente los pesos "
        "inversamente proporcional a la frecuencia de cada clase. Útil si "
        "hay desbalance (69.4% vs 30.6% en este dataset)."
    ),
    'bootstrap': (
        "Si se usa muestreo con reposición (bagging). True es el estándar "
        "y contribuye a la robustez del ensemble."
    ),
}

# Mostrar los 3+ hiperparámetros principales seleccionados
parametros_principales = ['n_estimators', 'max_depth', 'min_samples_split',
                          'min_samples_leaf', 'max_features']
count = 0
for param_name in parametros_principales:
    full_key = f'classifier__{param_name}'
    if full_key in best_params:
        count += 1
        value = best_params[full_key]
        just = justificaciones.get(param_name, "Sin justificación disponible")
        print(f"  {count}. {param_name} = {value}")
        print(f"     Justificación: {just}")
        print()

# =========================================================================
# 6. EVALUACIÓN CON VALIDACIÓN CRUZADA
# =========================================================================
print("\n[5/7] Evaluando modelos con validación cruzada estratificada...")

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
}

# Validación cruzada para Random Forest (mejor modelo)
cv_results_rf = cross_validate(
    search_rf.best_estimator_, X_train, y_train,
    cv=cv, scoring=scoring, return_train_score=True
)

# Validación cruzada para Regresión Logística (línea base)
cv_results_lr = cross_validate(
    pipeline_lr, X_train, y_train,
    cv=cv, scoring=scoring, return_train_score=True
)

print("\n  RESULTADOS DE VALIDACIÓN CRUZADA (5-Fold Stratified)")
print("  " + "=" * 65)
print(f"  {'Métrica':<20} {'Random Forest':>20} {'Reg. Logística':>20}")
print(f"  {'─' * 65}")

for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    rf_mean = cv_results_rf[f'test_{metric}'].mean()
    rf_std = cv_results_rf[f'test_{metric}'].std()
    lr_mean = cv_results_lr[f'test_{metric}'].mean()
    lr_std = cv_results_lr[f'test_{metric}'].std()
    print(f"  {metric:<20} {rf_mean:.4f} ± {rf_std:.4f} {lr_mean:.4f} ± {lr_std:.4f}")

print(f"  {'─' * 65}")

# Ganador
rf_f1 = cv_results_rf['test_f1'].mean()
lr_f1 = cv_results_lr['test_f1'].mean()
if rf_f1 > lr_f1:
    ganador = "Random Forest"
    ganador_f1 = rf_f1
    perdedor = "Regresión Logística"
    perdedor_f1 = lr_f1
else:
    ganador = "Regresión Logística"
    ganador_f1 = lr_f1
    perdedor = "Random Forest"
    perdedor_f1 = rf_f1

mejora_pct = ((ganador_f1 - perdedor_f1) / perdedor_f1) * 100
print(f"\n  GANADOR: {ganador} (F1 = {ganador_f1:.4f})")
print(f"  vs {perdedor} (F1 = {perdedor_f1:.4f})")
print(f"  Mejora relativa: {mejora_pct:.2f}%")

# =========================================================================
# 7. EVALUACIÓN FINAL EN CONJUNTO DE TEST (UNA SOLA VEZ)
# =========================================================================
print("\n[6/7] Evaluación final en conjunto de TEST (una sola vez)...")
print("  " + "=" * 50)

best_model = search_rf.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Matriz de confusión
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Métricas
accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)  # Recall
specificity = tn / (tn + fp)
ppv = tp / (tp + fp)  # Precision
npv = tn / (tn + fn)
f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity)
auc = roc_auc_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)

# Intervalos de confianza por bootstrap
n_bootstrap = 1000
np.random.seed(RANDOM_STATE)
auc_scores = []
f1_scores = []
y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
y_pred_np = y_pred.values if hasattr(y_pred, 'values') else y_pred
y_prob_np = y_prob.values if hasattr(y_prob, 'values') else y_prob
for _ in range(n_bootstrap):
    idx = np.random.choice(len(y_test_np), len(y_test_np), replace=True)
    auc_scores.append(roc_auc_score(y_test_np[idx], y_prob_np[idx]))
    f1_scores.append(f1_score(y_test_np[idx], y_pred_np[idx]))

auc_ic_lower = np.percentile(auc_scores, 2.5)
auc_ic_upper = np.percentile(auc_scores, 97.5)
f1_ic_lower = np.percentile(f1_scores, 2.5)
f1_ic_upper = np.percentile(f1_scores, 97.5)

print(f"\n  MATRIZ DE CONFUSIÓN:")
print(f"  {'':>15} {'Pred: Negativo':>15} {'Pred: Positivo':>15}")
print(f"  {'Real: Negativo':>15} {tn:>15} {fp:>15}")
print(f"  {'Real: Positivo':>15} {fn:>15} {tp:>15}")

print(f"\n  MÉTRICAS EN TEST:")
print(f"  {'─' * 50}")
print(f"  Accuracy (Exactitud):     {accuracy:.4f}")
print(f"  Sensibilidad (Recall):    {sensitivity:.4f}")
print(f"  Especificidad:            {specificity:.4f}")
print(f"  Precisión (PPV):          {ppv:.4f}")
print(f"  NPV:                      {npv:.4f}")
print(f"  F1-Score:                 {f1:.4f} (IC 95%: {f1_ic_lower:.4f} - {f1_ic_upper:.4f})")
print(f"  ROC-AUC:                  {auc:.4f} (IC 95%: {auc_ic_lower:.4f} - {auc_ic_upper:.4f})")
print(f"  Brier Score:              {brier:.4f}")
print(f"  {'─' * 50}")

# Umbral utilizado
print(f"\n  Umbral de clasificación: 0.5 (default)")
print(f"  Nota: El umbral puede ajustarse según criterio clínico")
print(f"  (priorizar sensibilidad para tamizaje o especificidad para confirmación)")

# Reporte completo de classification_report
print(f"\n  REPORTE DE CLASIFICACIÓN:")
print(classification_report(y_test, y_pred, target_names=['Sin diabetes', 'Diabetes']))

# =========================================================================
# 8. IMPORTANCIA DE VARIABLES
# =========================================================================
print("\n[7/7] Importancia de variables (Feature Importance):")
print("  " + "=" * 50)

rf_model = best_model.named_steps['classifier']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

for i, idx in enumerate(indices):
    bar = "█" * int(importances[idx] * 50)
    print(f"  {i+1}. {feature_names[idx]:<30} {importances[idx]:.4f} {bar}")

# =========================================================================
# 9. SERIALIZACIÓN DEL PIPELINE COMPLETO
# =========================================================================
print("\n" + "=" * 70)
print("SERIALIZACIÓN")
print("=" * 70)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Guardar pipeline completo (imputer + scaler + modelo)
pipeline_path = os.path.join(OUTPUT_DIR, 'diabetes_pipeline.pkl')
joblib.dump(best_model, pipeline_path)
print(f"\n  Pipeline guardado en: {pipeline_path}")
print(f"  Tamaño: {os.path.getsize(pipeline_path) / 1024:.1f} KB")

# Guardar también el modelo y scaler por separado (compatibilidad con app.py)
model_path = os.path.join(OUTPUT_DIR, 'random_forest_model.pkl')
scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
joblib.dump(best_model.named_steps['classifier'], model_path)
joblib.dump(best_model.named_steps['scaler'], scaler_path)
print(f"\n  Modelo separado: {model_path}")
print(f"  Scaler separado: {scaler_path}")

# =========================================================================
# 10. RESUMEN PARA EL INFORME
# =========================================================================
print("\n" + "=" * 70)
print("RESUMEN PARA EL INFORME DE TESIS")
print("=" * 70)
print(f"""
DATOS:
  - Dataset: Diabetes_prediction.csv ({len(df)} registros)
  - Distribución: {df['Diagnosis'].value_counts()[0]} negativos, {df['Diagnosis'].value_counts()[1]} positivos
  - División: Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}

HIPERPARÁMETROS AJUSTADOS (RandomizedSearchCV, {N_ITER_SEARCH} iteraciones, {CV_FOLDS}-Fold CV):
  1. n_estimators = {best_params.get('classifier__n_estimators', 'N/A')}
  2. max_depth = {best_params.get('classifier__max_depth', 'N/A')}
  3. min_samples_split = {best_params.get('classifier__min_samples_split', 'N/A')}""")

for p in ['min_samples_leaf', 'max_features', 'criterion', 'class_weight', 'bootstrap']:
    key = f'classifier__{p}'
    if key in best_params:
        print(f"  4. {p} = {best_params[key]}")
        break

print(f"""
RESULTADOS EN TEST:
  - Accuracy: {accuracy:.4f}
  - Sensibilidad: {sensitivity:.4f}
  - Especificidad: {specificity:.4f}
  - Precisión (PPV): {ppv:.4f}
  - F1-Score: {f1:.4f} (IC 95%: {f1_ic_lower:.4f} - {f1_ic_upper:.4f})
  - ROC-AUC: {auc:.4f} (IC 95%: {auc_ic_lower:.4f} - {auc_ic_upper:.4f})
  - Brier Score: {brier:.4f}

COMPARACIÓN CON LÍNEA BASE:
  - Random Forest F1: {rf_f1:.4f}
  - Regresión Logística F1: {lr_f1:.4f}
  - Mejora relativa: {mejora_pct:.2f}%

MÉTODO: Validación cruzada estratificada ({CV_FOLDS}-Fold) + RandomizedSearchCV ({N_ITER_SEARCH} iteraciones)
SEMILLA: {RANDOM_STATE}
LIBRERÍAS: scikit-learn {__import__('sklearn').__version__}, numpy {np.__version__}, pandas {pd.__version__}
""")

print("✅ Entrenamiento completado exitosamente.")
print(f"   Pipelines guardados en: {OUTPUT_DIR}")
