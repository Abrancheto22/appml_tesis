import pandas as pd
import numpy as np
from io import StringIO
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score,
    precision_score, f1_score, roc_auc_score,
    matthews_corrcoef, brier_score_loss
)
from scipy import stats

raw_data = """PACIENTE	PRETEST	POSTEST_SOFTWARE	POSTEST_BIN_SPSS	TP	TN	FP	FN
1	0	0,27	0	0	1	0	0
2	1	0,54	0	0	0	0	1
3	0	0,21	0	0	1	0	0
4	1	0,84	1	1	0	0	0
5	0	0,12	0	0	1	0	0
6	1	0,88	1	1	0	0	0
7	0	0,04	0	0	1	0	0
8	0	0,03	0	0	1	0	0
9	1	0,59	1	1	0	0	0
10	0	0	0	0	1	0	0
11	0	0,03	0	0	1	0	0
12	1	0,28	0	0	0	0	1
13	0	0,04	0	0	1	0	0
14	1	0,36	0	0	0	0	1
15	1	0,06	0	0	0	0	1
16	0	0,9	1	0	0	1	0
17	1	0,08	0	0	0	0	1
18	1	0,92	1	1	0	0	0
19	1	0,62	1	1	0	0	0
20	0	0,71	1	0	0	1	0
21	1	0,01	0	0	0	0	1
22	0	0,03	0	0	1	0	0
23	1	0,25	0	0	0	0	1
24	0	0,06	0	0	1	0	0
25	0	0,23	0	0	1	0	0
26	1	0,87	1	1	0	0	0
27	0	0,01	0	0	1	0	0
28	1	0,82	1	1	0	0	0
29	1	0,76	1	1	0	0	0
30	0	0,19	0	0	1	0	0
31	0	0,14	0	0	1	0	0
32	1	0,26	0	0	0	0	1
33	0	0,19	0	0	1	0	0
34	1	0,72	1	1	0	0	0
35	1	0,5	0	0	0	0	1
36	0	0,03	0	0	1	0	0
37	0	0,05	0	0	1	0	0
38	0	0,01	0	0	1	0	0
39	0	0,01	0	0	1	0	0
40	0	0,03	0	0	1	0	0
41	1	0,95	1	1	0	0	0
42	1	0,02	0	0	0	0	1
43	1	0,03	0	0	0	0	1
44	0	0,3	0	0	1	0	0
45	0	0,15	0	0	1	0	0
46	0	0,2	0	0	1	0	0
47	0	0,02	0	0	1	0	0
48	1	0,9	1	1	0	0	0
49	1	0,77	1	1	0	0	0
50	0	0,07	0	0	1	0	0
51	1	0,82	1	1	0	0	0
52	0	0	0	0	1	0	0
53	0	0,92	1	0	0	1	0
54	1	0,31	0	0	0	0	1
55	1	0,2	0	0	0	0	1
56	0	0,05	0	0	1	0	0
57	1	0,73	1	1	0	0	0
58	1	0,88	1	1	0	0	0
59	0	0,07	0	0	1	0	0
60	1	0,05	0	0	0	0	1
61	1	0,07	0	0	0	0	1
62	0	0,32	0	0	1	0	0
63	0	0,75	1	0	0	1	0
64	0	0,08	0	0	1	0	0
65	0	0,02	0	0	1	0	0
66	1	0,73	1	1	0	0	0
67	0	0,07	0	0	1	0	0
68	0	0	0	0	1	0	0
69	1	0,85	1	1	0	0	0
70	1	0,75	1	1	0	0	0
71	1	0,05	0	0	0	0	1
72	0	0,01	0	0	1	0	0
73	1	0,2	0	0	0	0	1
74	0	0,01	0	0	1	0	0
75	0	0,06	0	0	1	0	0
76	0	0,05	0	0	1	0	0
77	0	0,71	1	0	0	1	0
78	1	0,16	0	0	0	0	1
79	0	0,76	1	0	0	1	0
80	1	0,86	1	1	0	0	0"""

df = pd.read_csv(StringIO(raw_data), sep='\t')
df['POSTEST_SOFTWARE'] = df['POSTEST_SOFTWARE'].str.replace(',', '.').astype(float)

# Sumas exactas de tu Excel
tp_sum = df['TP'].sum()
tn_sum = df['TN'].sum()
fp_sum = df['FP'].sum()
fn_sum = df['FN'].sum()
total = len(df)

y_true = df['PRETEST'].values
y_pred_bin = df['POSTEST_BIN_SPSS'].values
y_prob = df['POSTEST_SOFTWARE'].values

# Métricas sobre el umbral 0.55 (Tu Excel)
acc_55 = (tp_sum + tn_sum) / total
sens_55 = tp_sum / (tp_sum + fn_sum)
spec_55 = tn_sum / (tn_sum + fp_sum)
prec_55 = tp_sum / (tp_sum + fp_sum)
npv_55 = tn_sum / (tn_sum + fn_sum)
f1_55 = 2 * (prec_55 * sens_55) / (prec_55 + sens_55)
auc_val = roc_auc_score(y_true, y_prob)
mcc_55 = matthews_corrcoef(y_true, y_pred_bin)
brier_55 = brier_score_loss(y_true, y_prob)

print(f"--- RESULTADOS EXACTOS DE TU TABLA EXCEL (N = {total} PACIENTES) ---")
print(f"Total Diabéticos Reales (PRETEST=1): {y_true.sum()} ({y_true.mean()*100:.2f}%)")
print(f"Total Sanos Reales (PRETEST=0):      {total - y_true.sum()} ({(1-y_true.mean())*100:.2f}%)")
print(f"Total Predichos Diabéticos (BIN=1):  {y_pred_bin.sum()} ({y_pred_bin.mean()*100:.2f}%)")
print(f"\nMatriz de Confusión en tus 80 pacientes (con umbral 0.55):")
print(f"  TP = {tp_sum}")
print(f"  TN = {tn_sum}")
print(f"  FP = {fp_sum}")
print(f"  FN = {fn_sum}")
print(f"\nMétricas Diagnósticas Reales de tus 80 pacientes:")
print(f"  Exactitud (Accuracy):  {acc_55*100:.2f}% ({tp_sum + tn_sum}/{total} aciertos)")
print(f"  Sensibilidad (Recall): {sens_55*100:.2f}% (detectó {tp_sum} de {tp_sum + fn_sum} diabéticos)")
print(f"  Especificidad:         {spec_55*100:.2f}% (descartó {tn_sum} de {tn_sum + fp_sum} sanos)")
print(f"  Precisión (PPV):       {prec_55*100:.2f}% ({tp_sum} de {tp_sum + fp_sum} positivos fueron correctos)")
print(f"  Valor Pred. Neg (NPV): {npv_55*100:.2f}%")
print(f"  F1-Score:              {f1_55:.4f}")
print(f"  ROC-AUC:               {auc_val:.4f}")
print(f"  MCC (Matthews):        {mcc_55:.4f} ({mcc_55*100:.2f}%)")
print(f"  Brier Score:           {brier_55:.4f}")

# McNemar entre Realidad (PRETEST) y Predicción (POSTEST_BIN)
# Tabla 2x2:
# a = (y_true=1 & y_pred=1) = TP = 19
# b = (y_true=0 & y_pred=1) = FP = 6
# c = (y_true=1 & y_pred=0) = FN = 18
# d = (y_true=0 & y_pred=0) = TN = 37
# McNemar b vs c:
b_mcnemar = fp_sum
c_mcnemar = fn_sum
chi2_mcn = ((abs(b_mcnemar - c_mcnemar) - 1)**2) / (b_mcnemar + c_mcnemar)
p_mcn = 1.0 - stats.chi2.cdf(chi2_mcn, df=1)

print(f"\nPrueba de McNemar en SPSS (Tabla 2x2):")
print(f"  b (FP) = {b_mcnemar}")
print(f"  c (FN) = {c_mcnemar}")
print(f"  Chi-cuadrado: {chi2_mcn:.4f}")
print(f"  p-valor SPSS: {p_mcn:.4f} (Sig = 0.023 en SPSS con prueba binomial/exacta!)")
