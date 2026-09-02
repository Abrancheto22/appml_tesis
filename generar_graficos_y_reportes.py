import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from generar_reporte_tesis import m_pre, m_post, y_test

# 1. Imprimir en terminal exactamente con el formato de las imágenes del usuario:
print("=" * 70)
print("--- Iniciando Evaluación del Modelo (Pretest - Modelo Base) ---")
print(f"Precisión (Accuracy) del modelo: {m_pre['Accuracy']:.4f}\n")
print("Reporte de Clasificación:")
print(classification_report(y_test, m_pre['pred'], digits=2))
print(f"Área Bajo la Curva (AUC): {m_pre['AUC']:.4f}")
print("=" * 70)

print("\n" + "=" * 70)
print("--- Iniciando Evaluación del Modelo (Postest - Modelo Optimizado) ---")
print(f"Precisión (Accuracy) del modelo: {m_post['Accuracy']:.4f}\n")
print("Reporte de Clasificación:")
print(classification_report(y_test, m_post['pred'], target_names=['Clase 0 (No Diabetes)', 'Clase 1 (Diabetes)'], digits=2))
print(f"Área Bajo la Curva (AUC): {m_post['AUC']:.4f}")
print("=" * 70)

# 2. Generar el gráfico exacto de la Matriz de Confusión (como la imagen 3)
cm = confusion_matrix(y_test, m_post['pred'])
plt.figure(figsize=(6, 5), dpi=300)
sns.set_theme(style="white")

ax = sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=True,
    xticklabels=['No Diabetes', 'Sí Diabetes'],
    yticklabels=['No Diabetes', 'Sí Diabetes'],
    annot_kws={"size": 13, "weight": "bold"}
)

plt.title('Matriz de Confusión', fontsize=14, pad=12, fontweight='bold')
plt.xlabel('Predicción del modelo', fontsize=11, labelpad=8)
plt.ylabel('Valor Real', fontsize=11, labelpad=8)
plt.tight_layout()

img_path = 'matriz_confusion.png'
plt.savefig(img_path, dpi=300)
plt.close()
print(f"\n✅ Gráfico de Matriz de Confusión generado exitosamente: {img_path}")
