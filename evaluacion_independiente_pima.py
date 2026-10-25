"""Evaluación exploratoria sin fuga de información sobre el PIMA original.

Este script no genera ni sobrescribe artefactos del modelo desplegado. Separa
los datos originales antes de ajustar la imputación y el escalado, mantiene
la prevalencia original en validación/prueba y fija el umbral en 0.5 antes de
ver los resultados. Sirve para detectar el efecto del flujo anterior con
SMOTE antes de la partición; no constituye validación clínica externa.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_OF_DEVELOPMENT = 0.25  # 20% del total
THRESHOLD = 0.5  # fijado antes de evaluar; no se ajusta con el test
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float | list]:
    predictions = (probabilities >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "sensitivity_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision_ppv": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
    }


def main() -> None:
    data_path = Path(__file__).with_name("pima_diabetes.csv")
    data = pd.read_csv(data_path)
    if "Outcome" not in data.columns:
        raise ValueError("El archivo PIMA debe tener la columna Outcome.")
    missing = sorted(set(FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Faltan variables requeridas: {missing}")

    X = data[FEATURES].copy()
    y = pd.to_numeric(data["Outcome"], errors="raise").astype(int)
    for column in ZERO_AS_MISSING:
        X.loc[X[column] == 0, column] = np.nan

    # Separar antes de cualquier fit/fit_resample; nada se balancea en val/test.
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=VALIDATION_SIZE_OF_DEVELOPMENT,
        random_state=RANDOM_STATE,
        stratify=y_development,
    )

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_leaf=8,
                    min_samples_split=16,
                    max_features="sqrt",
                    criterion="entropy",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    print("PIMA base sin sobremuestreo; no se guarda ni reemplaza ningún modelo.")
    print(
        f"Filas: {len(data)} | Train: {len(X_train)} | "
        f"Validación: {len(X_validation)} | Test final: {len(X_test)}"
    )
    print(
        f"Edades observadas: {X['Age'].min():g}–{X['Age'].max():g}; "
        f"casos menores de 21: {int((X['Age'] < 21).sum())}"
    )
    print(f"Umbral fijado antes de evaluar: {THRESHOLD:.2f}")

    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    print("Validación:", metrics(y_validation, validation_probabilities))

    # El test se consulta solo una vez, después de fijar modelo y umbral.
    test_probabilities = model.predict_proba(X_test)[:, 1]
    print("Test reservado:", metrics(y_test, test_probabilities))
    print(
        "Límite: este resultado corresponde a una partición aleatoria del PIMA "
        "original; no demuestra validez en otra población ni reemplaza una "
        "validación clínica prospectiva."
    )


if __name__ == "__main__":
    main()
