"""
ft_engineering.py
─────────────────────────────────────────────────────────────────
Feature Engineering Pipeline — Proyecto Créditos
Institución Financiera | Área de Datos y Analítica

Descripción:
    Primera componente del flujo de creación de modelos operativos.
    Genera los features transformados y retorna los conjuntos de
    datos de entrenamiento y evaluación listos para el modelado.

Salidas:
    - X_train, X_test, y_train, y_test (arrays numpy)
    - preprocessor (ColumnTransformer sklearn, serializado como .pkl)
    - feature_names (lista de nombres de columnas transformadas)
─────────────────────────────────────────────────────────────────
"""

import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─── Configuración de logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ─── Rutas ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH   = os.path.join(BASE_DIR, "Base_de_datos.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Constantes ──────────────────────────────────────────────────────────────
TARGET          = "Pago_atiempo"
TEST_SIZE       = 0.20
RANDOM_STATE    = 42
DATE_COL        = "fecha_prestamo"
FECHA_REF       = pd.Timestamp("2026-03-16")   # Fecha de referencia del proyecto

# Tipos de crédito válidos
TIPOS_CREDITO_VALIDOS = [4, 6, 7, 9, 10, 68]

# Categorías válidas de tendencia
TENDENCIA_CATS = ["Decreciente", "Estable", "Creciente"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA Y LIMPIEZA BASE
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Carga el dataset desde CSV y realiza parse de fechas."""
    log.info(f"Cargando datos desde: {path}")
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    log.info(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza y corrección de tipos de datos.
    Reglas de validación detectadas en el EDA.
    """
    log.info("Iniciando limpieza de datos...")
    df = df.copy()

    # ── 1. Unificar nulos en tendencia_ingresos ────────────────────────────
    df["tendencia_ingresos"] = df["tendencia_ingresos"].apply(
        lambda x: x if x in TENDENCIA_CATS else np.nan
    )

    # ── 2. Capping de outliers extremos ───────────────────────────────────
    # edad_cliente: max válido 90 años
    df["edad_cliente"] = df["edad_cliente"].clip(upper=90)

    # puntaje: no puede ser menor a -50 ni mayor a 100
    df["puntaje"] = df["puntaje"].clip(lower=-50, upper=100)

    # salario_cliente: capping en P99.5
    sal_cap = df["salario_cliente"].quantile(0.995)
    df["salario_cliente"] = df["salario_cliente"].clip(upper=sal_cap)

    # capital_prestado: capping en P99.5
    cap_cap = df["capital_prestado"].quantile(0.995)
    df["capital_prestado"] = df["capital_prestado"].clip(upper=cap_cap)

    # total_otros_prestamos: capping en P99
    top_cap = df["total_otros_prestamos"].quantile(0.99)
    df["total_otros_prestamos"] = df["total_otros_prestamos"].clip(upper=top_cap)

    # ── 3. Imputación simple antes de feature engineering ─────────────────
    df["saldo_mora_codeudor"] = df["saldo_mora_codeudor"].fillna(0.0)
    df["saldo_mora"] = df["saldo_mora"].fillna(0.0)
    df["saldo_total"] = df["saldo_total"].fillna(df["saldo_total"].median())
    df["saldo_principal"] = df["saldo_principal"].fillna(df["saldo_principal"].median())
    df["puntaje_datacredito"] = df["puntaje_datacredito"].fillna(df["puntaje_datacredito"].median())
    df["promedio_ingresos_datacredito"] = df["promedio_ingresos_datacredito"].fillna(0.0)

    log.info("Limpieza completada.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera atributos derivados identificados en el EDA.
    Estos features capturan relaciones no lineales y conceptos
    financieros clave como capacidad de pago y nivel de apalancamiento.
    """
    log.info("Generando features derivados...")
    df = df.copy()
    EPS = 1e-6  # Para evitar división por cero

    # ── Features temporales ────────────────────────────────────────────────
    df["mes_prestamo"]            = df[DATE_COL].dt.month
    df["trimestre_prestamo"]      = df[DATE_COL].dt.quarter
    df["anio_prestamo"]           = df[DATE_COL].dt.year
    df["dia_semana_prestamo"]     = df[DATE_COL].dt.dayofweek
    df["antiguedad_dias"]         = (FECHA_REF - df[DATE_COL]).dt.days.clip(lower=0)

    # ── Ratios financieros ─────────────────────────────────────────────────
    df["ratio_cuota_salario"]     = df["cuota_pactada"] / (df["salario_cliente"] + EPS)
    df["ratio_deuda_ingreso"]     = (df["total_otros_prestamos"] + df["capital_prestado"]) / \
                                     (df["salario_cliente"] + EPS)
    df["ratio_mora_saldo"]        = df["saldo_mora"] / (df["saldo_total"] + EPS)
    df["ratio_capital_salario"]   = df["capital_prestado"] / (df["salario_cliente"] + EPS)

    # ── Indicadores binarios ───────────────────────────────────────────────
    df["tiene_mora"]              = (df["saldo_mora"] > 0).astype(int)
    df["tiene_mora_codeudor"]     = (df["saldo_mora_codeudor"] > 0).astype(int)
    df["sin_datacredito"]         = df["tendencia_ingresos"].isna().astype(int)
    df["tipo_laboral_bin"]        = (df["tipo_laboral"] == "Empleado").astype(int)

    # ── Agregados ──────────────────────────────────────────────────────────
    df["total_creditos"]          = (df["creditos_sectorFinanciero"] +
                                     df["creditos_sectorCooperativo"] +
                                     df["creditos_sectorReal"])
    df["score_combinado"]         = (df["puntaje"] + df["puntaje_datacredito"] / 10) / 2

    # ── Transformaciones logarítmicas (reducir asimetría) ─────────────────
    for col in ["capital_prestado", "salario_cliente", "cuota_pactada",
                "total_otros_prestamos", "promedio_ingresos_datacredito",
                "saldo_total", "saldo_principal"]:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # ── Variable de tendencia con NaN unificado ────────────────────────────
    df["tendencia_ingresos"] = df["tendencia_ingresos"].fillna("Sin_info")

    log.info(f"Features derivados creados. Nuevas dimensiones: {df.shape}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DEFINICIÓN DE COLUMNAS PARA EL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

# Columnas a excluir del modelado
EXCLUDE_COLS = [TARGET, DATE_COL, "saldo_total", "saldo_principal",
                "tipo_laboral",    # ya codificada como tipo_laboral_bin
                "anio_prestamo"]   # baja varianza (todos 2024-2026)

# Variables numéricas que van al scaler
NUMERIC_COLS = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "saldo_mora",
    "saldo_mora_codeudor", "creditos_sectorFinanciero",
    "creditos_sectorCooperativo", "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    # Derivados numéricos
    "ratio_cuota_salario", "ratio_deuda_ingreso", "ratio_mora_saldo",
    "ratio_capital_salario", "total_creditos", "score_combinado",
    "antiguedad_dias", "mes_prestamo", "trimestre_prestamo", "dia_semana_prestamo",
    "log_capital_prestado", "log_salario_cliente", "log_cuota_pactada",
    "log_total_otros_prestamos", "log_promedio_ingresos_datacredito",
    "log_saldo_total", "log_saldo_principal",
    # Indicadores binarios
    "tiene_mora", "tiene_mora_codeudor", "sin_datacredito", "tipo_laboral_bin",
]

# Variables categóricas nominales → OneHotEncoding
CATEG_NOMINAL_COLS = ["tipo_credito"]

# Variables categóricas ordinales → OrdinalEncoding
CATEG_ORDINAL_COLS = ["tendencia_ingresos"]
ORDINAL_CATEGORIES = [["Decreciente", "Sin_info", "Estable", "Creciente"]]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CONSTRUCCIÓN DEL PREPROCESSOR (ColumnTransformer)
# ═══════════════════════════════════════════════════════════════════════════════

def build_preprocessor() -> ColumnTransformer:
    """
    Construye el ColumnTransformer que aplica las transformaciones
    correctas a cada tipo de variable, tal como se muestra en el
    diagrama del proyecto:

    ColumnTransformer
    ├── numeric      → SimpleImputer(median) → StandardScaler
    ├── categoric    → SimpleImputer(most_frequent) → OneHotEncoder
    └── ordinal      → SimpleImputer(most_frequent) → OrdinalEncoder
    """
    log.info("Construyendo ColumnTransformer (preprocessor)...")

    # Pipeline para numéricas
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])

    # Pipeline para categóricas nominales
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Pipeline para categóricas ordinales
    ordinal_pipeline = Pipeline(steps=[
        ("imputer",  SimpleImputer(strategy="most_frequent")),
        ("ordinal",  OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric",   numeric_pipeline,      NUMERIC_COLS),
        ("categoric", categorical_pipeline,  CATEG_NOMINAL_COLS),
        ("ordinal",   ordinal_pipeline,      CATEG_ORDINAL_COLS),
    ], remainder="drop")

    log.info("Preprocessor construido exitosamente.")
    return preprocessor


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PIPELINE PRINCIPAL DE FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def run_feature_engineering(
    data_path: str = DATA_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    save_artifacts: bool = True
):
    """
    Pipeline completo de Feature Engineering.

    Args:
        data_path      : Ruta al archivo CSV con datos crudos.
        test_size      : Proporción para conjunto de prueba (default 0.20).
        random_state   : Semilla para reproducibilidad.
        save_artifacts : Si True, guarda el preprocessor en disco.

    Returns:
        X_train (np.array), X_test (np.array),
        y_train (pd.Series), y_test (pd.Series),
        preprocessor (ColumnTransformer),
        feature_names (list[str])
    """
    # ── Paso 1: Carga ──────────────────────────────────────────────────────
    df = load_data(data_path)

    # ── Paso 2: Limpieza ───────────────────────────────────────────────────
    df = clean_data(df)

    # ── Paso 3: Features derivados ─────────────────────────────────────────
    df = add_derived_features(df)

    # ── Paso 4: Separar X e y ──────────────────────────────────────────────
    X = df[[c for c in NUMERIC_COLS + CATEG_NOMINAL_COLS + CATEG_ORDINAL_COLS
             if c in df.columns]]
    y = df[TARGET]

    log.info(f"X shape: {X.shape} | y distribución: {dict(y.value_counts())}")

    # ── Paso 5: Split estratificado ────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    log.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # ── Paso 6: Fit & Transform del preprocessor ───────────────────────────
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    # ── Paso 7: Obtener nombres de features ────────────────────────────────
    ohe_cats = preprocessor.named_transformers_["categoric"] \
                            .named_steps["onehot"] \
                            .get_feature_names_out(CATEG_NOMINAL_COLS).tolist()
    feature_names = NUMERIC_COLS + ohe_cats + CATEG_ORDINAL_COLS

    log.info(f"Features finales: {len(feature_names)}")
    log.info(f"X_train transformado: {X_train_t.shape}")

    # ── Paso 8: Guardar artefactos ─────────────────────────────────────────
    if save_artifacts:
        pp_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
        joblib.dump(preprocessor, pp_path)
        log.info(f"Preprocessor guardado en: {pp_path}")

        # Guardar conjuntos de datos procesados
        pd.DataFrame(X_train_t, columns=feature_names).to_csv(
            os.path.join(MODELS_DIR, "X_train.csv"), index=False)
        pd.DataFrame(X_test_t, columns=feature_names).to_csv(
            os.path.join(MODELS_DIR, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(MODELS_DIR, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(MODELS_DIR, "y_test.csv"), index=False)
        log.info("Conjuntos de datos guardados en /models/")

    log.info("✅ Feature Engineering completado.")
    return X_train_t, X_test_t, y_train, y_test, preprocessor, feature_names


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EJECUCIÓN DIRECTA
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("FEATURE ENGINEERING PIPELINE")
    log.info("=" * 60)

    X_train, X_test, y_train, y_test, preprocessor, feature_names = \
        run_feature_engineering()

    print("\n" + "─" * 60)
    print("RESUMEN DEL FEATURE ENGINEERING")
    print("─" * 60)
    print(f"  X_train shape   : {X_train.shape}")
    print(f"  X_test  shape   : {X_test.shape}")
    print(f"  y_train distrib : {dict(y_train.value_counts())}")
    print(f"  y_test  distrib : {dict(y_test.value_counts())}")
    print(f"  Nº Features     : {len(feature_names)}")
    print(f"  Top features    : {feature_names[:10]}")
    print("─" * 60)
    print("✅ Listo para entrenamiento.")
