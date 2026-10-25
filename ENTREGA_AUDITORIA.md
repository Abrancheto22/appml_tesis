# Entrega de auditoría: API Python/ML

Fecha: 2026-10-24

Esta rama contiene cambios preparados para revisión. El push a `main` publicaría
automáticamente el servicio en Vercel; por eso esta entrega usa una rama de
auditoría y no modifica Production.

## Trabajo realizado

- Endurecimiento del endpoint de predicción: requiere `ML_API_KEY` en
  `Authorization: Bearer ...` y no devuelve el payload recibido en modo debug.
- Validación de campos numéricos, límites configurables de edad, errores y
  respuesta del servicio.
- Actualización de dependencias y pin de scikit-learn compatible con el
  artefacto actual.
- Límites por defecto `ML_AGE_MIN=21` y `ML_AGE_MAX=81`.
- Guía de configuración/despliegue en `DEPLOY_VERCEL.md`.
- `VALIDACION_MODELO.md` resume métricas de tesis y límites metodológicos.
- `evaluacion_independiente_pima.py` evalúa de forma exploratoria separando
  primero los datos originales; no reemplaza el artefacto de producción.

## Modelo y límites conocidos

El modelo desplegable conserva las métricas reportadas en la tesis. La
evaluación independiente exploratoria obtuvo, con una prueba reservada de 154
registros, accuracy 75.3%, sensibilidad 74.1%, especificidad 76.0%, AUC 0.834 y
MCC 0.485. La muestra es pequeña y no constituye validación clínica externa.

El flujo histórico aplicó SMOTE antes de la división, con riesgo de fuga entre
particiones. No presentar esas métricas como estimación independiente del
desempeño en pacientes nuevos. No cambiar el modelo ni ampliar el rango de edad
sin nueva evaluación, aprobación del equipo/asesor y validación clínica.

## Pendiente antes de Production

1. Crear una clave aleatoria segura de 32+ caracteres y configurarla como
   secreto `ML_API_KEY` en Production de Vercel y en Laravel. No está incluida
   en Git. El código nuevo devuelve `401` si falta la clave.
2. Acordar el momento coordinado del despliegue con Laravel: la versión Laravel
   ya preparada también debe enviar la misma clave. Antes de cambiar Production,
   revisar que ambos servicios estén configurados para evitar interrupciones.
3. En Preview, probar `/` y verificar `pipeline_loaded: true`; confirmar `401`
   sin clave; con clave, validar una solicitud ficticia y límites 422 en edades
   fuera de la configuración. No usar datos identificables de pacientes.
4. Revisar logs de Vercel y confirmar que no se registren claves ni entradas
   clínicas. Consultar `DEPLOY_VERCEL.md` para las variables completas.

## Despliegue

El repositorio está conectado a Git → Vercel. Un push a la rama de Production
despliega automáticamente; un push de esta rama debe quedar como Preview según
la configuración del proyecto. Este handoff no ha cambiado secretos ni ha
activado Production.
