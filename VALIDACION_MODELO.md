# Estado de validación del modelo DT2

## Rango de edad del artefacto actual

El conjunto `dataset_clinico_10k.csv` tiene edades de 21 a 81 años. Las medias
del `StandardScaler` del artefacto `modelo/diabetes_pipeline.pkl` coinciden con
las del conjunto de entrenamiento de 6,000 filas obtenido al volver a dividir
ese CSV con la semilla 42. No hay observaciones menores de 21 en los datos de
origen.

Por eso la API y Laravel admiten por defecto `ML_AGE_MIN=21` y
`ML_AGE_MAX=81`. Son límites del soporte de los datos del modelo; no son una
afirmación sobre quién puede tener diabetes. Fuera de esos límites el sistema
no estima y comunica que eso no descarta la enfermedad.

Las guías clínicas consideran cribado de riesgo de diabetes tipo 2 desde la
pubertad o alrededor de los 10 años en jóvenes con sobrepeso/obesidad y otros
factores de riesgo. Esas recomendaciones no validan este clasificador para
menores: este sistema no recibe percentil de IMC ni todos los factores
pediátricos necesarios. La evaluación clínica debe continuar por la vía
correspondiente aunque no se muestre una predicción.

## Limitación metodológica encontrada

`evaluacion_completa_10k.py` imputa y aplica SMOTE a los datos PIMA antes de
partirlos. Luego `generar_reporte_realista_10k.py` vuelve a particionar el
conjunto ya sobremuestreado. Muestras sintéticas relacionadas pueden aparecer
en entrenamiento y prueba; además, la prevalencia de la prueba deja de
representar la población original. Por ello las métricas publicadas por ese
flujo no deben presentarse como estimación independiente del rendimiento en
pacientes nuevos. La documentación oficial de scikit-learn recomienda separar
primero y ajustar el preprocesamiento solo con entrenamiento; imbalanced-learn
advierte que sobremuestrear antes de dividir produce fuga y estimaciones
optimistas.

## Evaluación exploratoria sin fuga

Se añadió `evaluacion_independiente_pima.py`. Usa el archivo PIMA original,
separa los datos antes de ajustar imputación y escalado, conserva la
prevalencia original en validación y prueba, fija el umbral en 0.5 y no guarda
ni reemplaza el modelo desplegado.

Con la partición reproducible (semilla 42), los resultados exploratorios en
una prueba reservada de 154 registros fueron: accuracy 75.3%, sensibilidad
74.1%, especificidad 76.0%, AUC 0.834 y MCC 0.485. El pequeño tamaño de la
prueba hace que estas cifras tengan incertidumbre considerable; no son una
validación clínica externa ni deben sustituir por sí solas las métricas de la
tesis.

## Próximos pasos antes de cambiar el artefacto o desplegar

1. Acordar con el asesor y el equipo clínico el diseño estadístico y qué
   resultados se presentarán. No reutilizar como prueba independiente las
   métricas del flujo con SMOTE previo a la partición.
2. Entrenar y evaluar candidatos separando la población original primero;
   ajustar imputación, escalado, balanceo y selección de umbral solo dentro de
   entrenamiento/validación. Mantener una prueba natural reservada sin tocar.
3. Validar externamente con una población representativa del ámbito de uso,
   revisar calibración y sensibilidad, y definir supervisión clínica.
4. Solo tras esa aprobación, actualizar juntos el artefacto, las métricas de
   la ficha del modelo y los límites de edad; después seguir
   `DEPLOY_VERCEL.md`.

Para ampliar el uso a menores se necesita un conjunto pediátrico apropiado,
variables pediátricas (por ejemplo, percentil de IMC y factores de riesgo) y
validación específica. No se debe bajar `ML_AGE_MIN` solo con una variable de
entorno.
