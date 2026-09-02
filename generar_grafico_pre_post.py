import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Métricas Pretest vs Posttest
metricas = ['Exactitud\n(Accuracy)', 'Sensibilidad\n(Recall DT2)', 'Especificidad\n(No DT2)', 'F1-Score\n(Armónico)', 'ROC-AUC\n(Área Curva)', 'MCC\n(Matthews)']
pretest = [84.35, 91.50, 77.20, 85.39, 92.26, 69.41]
postest = [90.85, 95.50, 86.20, 91.26, 97.22, 82.06]

x = np.arange(len(metricas))
width = 0.36

plt.figure(figsize=(9.5, 5), dpi=300)
sns.set_theme(style="whitegrid")

rects1 = plt.bar(x - width/2, pretest, width, label='PRETEST (Modelo Base)', color='#4A5568')
rects2 = plt.bar(x + width/2, postest, width, label='POSTEST (Modelo Optimizado)', color='#3182CE')

plt.ylabel('Puntuación del Desempeño (%)', fontsize=11, fontweight='bold')
plt.title('Figura 6. Gráfico Comparativo PRETEST - POSTEST del Desempeño del Modelo de ML', fontsize=12, fontweight='bold', pad=14)
plt.xticks(x, metricas, fontsize=9.5, fontweight='bold')
plt.ylim(0, 115)

# Añadir etiquetas encima de las barras
for i in range(len(metricas)):
    plt.text(x[i] - width/2, pretest[i] + 1.5, f"{pretest[i]:.1f}%", ha='center', fontsize=8.5, fontweight='bold', color='#2D3748')
    plt.text(x[i] + width/2, postest[i] + 1.5, f"{postest[i]:.1f}%", ha='center', fontsize=9, fontweight='bold', color='#1A365D')

plt.legend(frameon=True, facecolor='white', loc='lower right', fontsize=9.5)
plt.tight_layout()
plt.savefig('grafico_pretest_postest_actualizado.png', dpi=300)
plt.close()
print("✅ Gráfico comparativo Pretest-Postest generado: grafico_pretest_postest_actualizado.png")
