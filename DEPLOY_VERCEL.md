# Configuración del modelo en Vercel

Esta guía cubre el servicio Flask/Python (`appml_tesis`) que entrega las
predicciones. El código actualizado exige una clave compartida en cada llamada
a `POST /predict`; por eso la variable debe coincidir con la configurada en
Laravel. Estos cambios todavía deben desplegarse para aplicarse al endpoint
remoto.

## 1. Crear una clave compartida

Genera una clave aleatoria de al menos 32 caracteres en un gestor seguro o con
un generador criptográfico local. No la incluyas en el repositorio, en
JavaScript del navegador, ni en mensajes o capturas. La misma clave se debe
guardar como secreto en Vercel y en el `.env` del servidor Laravel.

## 2. Revisar el proyecto de Vercel

En **Project Settings → Build and Deployment**, confirma que el **Root
Directory** sea `appml_tesis` (si el repositorio conectado es el directorio
padre). El proyecto ya incluye `vercel.json`, `requirements.txt` y
`.python-version`; el runtime Python toma la versión definida en
`.python-version` y las dependencias de `requirements.txt`.

En **Settings → Environment Variables**, agrega:

| Nombre | Valor | Entorno |
| --- | --- | --- |
| `ML_API_KEY` | La clave aleatoria generada | Production; Preview solo si se configura un cliente de pruebas con una clave distinta |
| `ML_AGE_MIN` | `21` | Production |
| `ML_AGE_MAX` | `81` | Production |
| `ALLOWED_ORIGINS` | No agregar para llamadas servidor a servidor. Configurar solo si un navegador invoca directamente la API | Solo si aplica |

`ALLOWED_ORIGINS` solo debe contener orígenes exactos, separados por comas,
si un navegador llama directamente a la API. No uses `*`. Para el flujo
Laravel → Python, CORS no es necesario.

El rango `21–81` es el intervalo observado en `dataset_clinico_10k.csv`, que
coincide con las medias del conjunto usado para ajustar el pipeline activo.
Mantén iguales los valores `ML_AGE_MIN` y `ML_AGE_MAX` en Vercel y Laravel.
Puedes estrechar el intervalo configurado dentro de `21–81`; el servicio
rechaza por configuración cualquier intento de ampliarlo. Para ampliar el
rango primero se necesita reentrenar y validar un nuevo modelo con datos
representativos de la nueva población. Este rango
describe dónde hay edades en los datos del proyecto; por sí solo no demuestra
validez clínica para ninguna edad.

**Limitación metodológica detectada:** `dataset_clinico_10k.csv` se generó
aplicando SMOTE a los 768 registros Pima antes de dividir en entrenamiento,
validación y prueba. Por ello pueden quedar muestras sintéticas relacionadas
entre particiones; las métricas reportadas no deben describirse como una prueba
independiente hasta rehacer la evaluación separando primero la muestra original
y aplicando el balanceo solo dentro del entrenamiento.

Esto es un límite del soporte de **este modelo**, no una edad por debajo de la
cual sea imposible tener diabetes. ADA e ISPAD recomiendan considerar el
cribado de riesgo de tipo 2 desde la pubertad o alrededor de los 10 años en
jóvenes con sobrepeso/obesidad y factores adicionales; ISPAD contempla grupos
de riesgo alto incluso antes. Estas recomendaciones clínicas no validan este
clasificador en menores. Fuentes: [ADA, Standards of Care 2026](https://diabetesjournals.org/care/article/49/Supplement_1/S297/163924/14-Children-and-Adolescents-Standards-of-Care-in)
y [ISPAD, guía clínica 2024 sobre tipo 2 en niños y adolescentes](https://pmc.ncbi.nlm.nih.gov/articles/PMC11854986/).

## 3. Desplegar

Una vez guardadas las variables, crea un nuevo despliegue de Production desde
Vercel (por la integración Git o el botón **Redeploy**). Los cambios de
variables no se aplican a despliegues anteriores. Revisa el build log y
confirma que la función se construya con el pipeline y sus dependencias.

## 4. Configurar el cliente Laravel

En el `.env` local de Laravel agrega la misma clave, sin comillas adicionales:

```dotenv
ML_API_URL=https://appml-tesis.vercel.app/predict
ML_API_KEY=<la-misma-clave-guardada-en-Vercel>
ML_AGE_MIN=21
ML_AGE_MAX=81
```

Después de cambiar variables de Laravel, limpia su caché de configuración con
`php artisan config:clear`. Si Laravel se despliega posteriormente, configura
ambas variables en el entorno secreto del hosting de Laravel y reinicia o
redespliega ese servicio. Nunca uses un prefijo `VITE_` para la clave.

## 5. Comprobar el despliegue

1. Abre `https://appml-tesis.vercel.app/` y confirma que la respuesta indique
   `pipeline_loaded: true`.
2. Envía `POST /predict` sin `Authorization`: debe responder `401`.
3. Envía una predicción de prueba con
   `Authorization: Bearer <clave>` y los ocho campos numéricos requeridos: debe
   responder `200` con las probabilidades, sin repetir los datos recibidos.
4. Envía una variable fuera de rango: debe responder `422`.
5. Repite la predicción desde Laravel y confirma que se guarda correctamente.

Como comprobación de límites, las edades 20 y 82 deben devolver `422`; 21 y
81 están en los extremos admitidos por la configuración actual. Un `422` por
edad fuera de rango significa que el modelo no tiene datos para esa edad,
nunca que la persona no tenga diabetes. Además, la guía pediátrica evalúa
percentil de IMC y factores de riesgo, que este sistema no recibe.

No uses datos identificables ni registros reales de pacientes para estas
comprobaciones. Si la API devuelve `401`, confirma que Laravel y Vercel tengan
exactamente la misma clave y que el despliegue nuevo ya esté activo. Para
rotarla, actualiza ambos entornos y despliega de nuevo; una diferencia temporal
entre las claves interrumpirá las predicciones.

Referencias oficiales: [runtime Python de Vercel](https://vercel.com/docs/functions/runtimes/python)
y [gestión de variables de entorno](https://vercel.com/docs/environment-variables/managing-environment-variables).
