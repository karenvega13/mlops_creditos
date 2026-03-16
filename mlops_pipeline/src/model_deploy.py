"""
model_deploy.py
─────────────────────────────────────────────────────────────────
Despliegue del Modelo como API REST — Proyecto Créditos
Institución Financiera | Área de Datos y Analítica

Descripción:
    Expone el mejor modelo entrenado como un servicio REST
    usando FastAPI. Soporta predicción individual y por lotes (batch).

Endpoints:
    GET  /          → Health check
    GET  /info      → Información del modelo en producción
    POST /predict   → Predicción de un único registro
    POST /predict/batch → Predicción por lote (múltiples registros)

Ejecutar:
    uvicorn model_deploy:app --host 0.0.0.0 --port 8000 --reload

Docker:
    docker build -t mlops-creditos:latest .
    docker run -p 8000:8000 mlops-creditos:latest
─────────────────────────────────────────────────────────────────
"""

import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
import joblib
from typing import List, Optional, Union
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

warnings.filterwarnings("ignore")

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ─── Rutas ───────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR        = os.path.join(BASE_DIR, "models")
MODEL_PATH        = os.path.join(MODELS_DIR, "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")

sys.path.insert(0, os.path.dirname(__file__))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA DEL MODELO
# ═══════════════════════════════════════════════════════════════════════════════

def load_artifacts():
    """Carga el modelo y preprocesador desde disco."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modelo no encontrado en {MODEL_PATH}. "
            f"Ejecute primero model_training_evaluation.py"
        )
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"Preprocesador no encontrado en {PREPROCESSOR_PATH}."
        )

    model        = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    log.info(f"✅ Modelo cargado: {MODEL_PATH}")
    log.info(f"✅ Preprocesador cargado: {PREPROCESSOR_PATH}")
    return model, preprocessor


# Cargar al iniciar la app
try:
    MODEL, PREPROCESSOR = load_artifacts()
    MODEL_LOADED = True
    MODEL_INFO = {
        "model_type"   : type(MODEL).__name__,
        "loaded_at"    : datetime.now().isoformat(),
        "model_path"   : MODEL_PATH,
        "version"      : "1.1.0"
    }
except Exception as e:
    log.warning(f"Modelo no disponible: {e}")
    MODEL, PREPROCESSOR = None, None
    MODEL_LOADED = False
    MODEL_INFO = {}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ESQUEMAS DE ENTRADA/SALIDA (Pydantic)
# ═══════════════════════════════════════════════════════════════════════════════

class CreditRecord(BaseModel):
    """
    Esquema de un registro crediticio para predicción.
    Todos los campos corresponden a las variables del dataset original.
    """
    tipo_credito              : int   = Field(..., ge=4, le=68,       description="Código tipo de crédito")
    fecha_prestamo            : str   = Field(...,                     description="Fecha de desembolso (YYYY-MM-DD)")
    capital_prestado          : float = Field(..., ge=0,               description="Monto del crédito en COP")
    plazo_meses               : int   = Field(..., ge=1, le=120,       description="Plazo en meses")
    edad_cliente              : int   = Field(..., ge=18, le=90,       description="Edad del cliente")
    tipo_laboral              : str   = Field(...,                     description="Empleado o Independiente")
    salario_cliente           : float = Field(..., ge=0,               description="Ingreso mensual en COP")
    total_otros_prestamos     : float = Field(..., ge=0,               description="Saldo total otros préstamos")
    cuota_pactada             : float = Field(..., ge=0,               description="Cuota mensual pactada")
    puntaje                   : float = Field(...,                     description="Puntaje interno de riesgo")
    puntaje_datacredito       : float = Field(None,                    description="Score Datacrédito")
    cant_creditosvigentes     : int   = Field(..., ge=0,               description="Créditos vigentes en el sistema")
    huella_consulta           : int   = Field(..., ge=0,               description="Consultas en centrales de riesgo")
    saldo_mora                : float = Field(0.0, ge=0,               description="Saldo en mora")
    saldo_total               : float = Field(..., ge=0,               description="Saldo total obligación")
    saldo_principal           : float = Field(..., ge=0,               description="Saldo de capital")
    saldo_mora_codeudor       : float = Field(0.0, ge=0,               description="Mora del codeudor (0 si no aplica)")
    creditos_sectorFinanciero : int   = Field(..., ge=0)
    creditos_sectorCooperativo: int   = Field(..., ge=0)
    creditos_sectorReal       : int   = Field(..., ge=0)
    promedio_ingresos_datacredito: Optional[float] = Field(None)
    tendencia_ingresos        : Optional[str] = Field(None,            description="Estable/Creciente/Decreciente")

    @validator("tipo_laboral")
    def validate_tipo_laboral(cls, v):
        if v not in ["Empleado", "Independiente"]:
            raise ValueError("tipo_laboral debe ser 'Empleado' o 'Independiente'")
        return v

    @validator("tendencia_ingresos")
    def validate_tendencia(cls, v):
        valid = ["Estable", "Creciente", "Decreciente", None]
        if v not in valid:
            raise ValueError(f"tendencia_ingresos debe ser uno de: {valid}")
        return v

    class Config:
        schema_extra = {
            "example": {
                "tipo_credito": 4,
                "fecha_prestamo": "2025-03-15",
                "capital_prestado": 2000000,
                "plazo_meses": 12,
                "edad_cliente": 35,
                "tipo_laboral": "Empleado",
                "salario_cliente": 3500000,
                "total_otros_prestamos": 500000,
                "cuota_pactada": 185000,
                "puntaje": 95.22,
                "puntaje_datacredito": 780.0,
                "cant_creditosvigentes": 3,
                "huella_consulta": 2,
                "saldo_mora": 0.0,
                "saldo_total": 2000000,
                "saldo_principal": 1950000,
                "saldo_mora_codeudor": 0.0,
                "creditos_sectorFinanciero": 2,
                "creditos_sectorCooperativo": 0,
                "creditos_sectorReal": 1,
                "promedio_ingresos_datacredito": 3200000.0,
                "tendencia_ingresos": "Estable"
            }
        }


class BatchRequest(BaseModel):
    """Solicitud de predicción por lote."""
    records : List[CreditRecord]
    threshold: float = Field(0.5, ge=0.0, le=1.0,
                              description="Umbral de decisión (default 0.5)")


class PredictionResponse(BaseModel):
    """Respuesta de predicción individual."""
    prediction        : int
    probability_default: float
    probability_ontime : float
    risk_label        : str
    timestamp         : str


class BatchResponse(BaseModel):
    """Respuesta de predicción por lote."""
    predictions : List[PredictionResponse]
    total       : int
    predicted_default: int
    predicted_ontime : int
    default_rate: float


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LÓGICA DE PREDICCIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_input(records: List[CreditRecord]) -> pd.DataFrame:
    """Convierte registros Pydantic a DataFrame y aplica feature engineering."""
    from ft_engineering import clean_data, add_derived_features, \
                               NUMERIC_COLS, CATEG_NOMINAL_COLS, CATEG_ORDINAL_COLS

    data = [r.dict() for r in records]
    df = pd.DataFrame(data)
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])

    df = clean_data(df)
    df = add_derived_features(df)

    feature_cols = [c for c in NUMERIC_COLS + CATEG_NOMINAL_COLS + CATEG_ORDINAL_COLS
                    if c in df.columns]
    return df[feature_cols]


def make_prediction(df_features: pd.DataFrame, threshold: float = 0.5):
    """Ejecuta la predicción sobre el DataFrame de features."""
    X = PREPROCESSOR.transform(df_features)
    probs = MODEL.predict_proba(X)
    preds = (probs[:, 1] >= threshold).astype(int)  # 1 = pagó a tiempo
    return preds, probs


def get_risk_label(prob_ontime: float) -> str:
    """Etiqueta de riesgo según probabilidad de pago a tiempo."""
    if prob_ontime >= 0.80:
        return "RIESGO BAJO"
    elif prob_ontime >= 0.60:
        return "RIESGO MEDIO"
    elif prob_ontime >= 0.40:
        return "RIESGO ALTO"
    else:
        return "RIESGO MUY ALTO"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. APLICACIÓN FASTAPI
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="API Modelo Crediticio — Pago a Tiempo",
    description="""
    API REST para predicción de comportamiento de pago de créditos.
    Retorna la probabilidad de que un cliente pague su crédito a tiempo.

    **Variable objetivo:** `Pago_atiempo` — 1 = Pagó a tiempo, 0 = No pagó
    """,
    version="1.1.0",
    contact={"name": "Equipo de Datos y Analítica"}
)


@app.get("/", tags=["Health"])
def health_check():
    """Verificación de estado del servicio."""
    return {
        "status"      : "healthy" if MODEL_LOADED else "model_not_loaded",
        "service"     : "Modelo Crediticio — Pago a Tiempo",
        "version"     : "1.1.0",
        "timestamp"   : datetime.now().isoformat(),
        "model_loaded": MODEL_LOADED
    }


@app.get("/info", tags=["Model"])
def model_info():
    """Información del modelo actualmente en producción."""
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. Ejecute el pipeline de entrenamiento."
        )
    return MODEL_INFO


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(record: CreditRecord, threshold: float = 0.5):
    """
    Predice el comportamiento de pago para un único registro.

    - **prediction**: 1 = Pagará a tiempo, 0 = No pagará a tiempo
    - **probability_ontime**: Probabilidad de pago a tiempo [0-1]
    - **probability_default**: Probabilidad de incumplimiento [0-1]
    - **risk_label**: Etiqueta de riesgo (BAJO / MEDIO / ALTO / MUY ALTO)
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    try:
        df_features = prepare_input([record])
        preds, probs = make_prediction(df_features, threshold)

        prob_default = float(probs[0][0])
        prob_ontime  = float(probs[0][1])
        prediction   = int(preds[0])

        log.info(f"Predicción: {prediction} | P(ontime)={prob_ontime:.4f}")

        return PredictionResponse(
            prediction        = prediction,
            probability_default = round(prob_default, 4),
            probability_ontime  = round(prob_ontime,  4),
            risk_label        = get_risk_label(prob_ontime),
            timestamp         = datetime.now().isoformat()
        )

    except Exception as e:
        log.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(request: BatchRequest):
    """
    Predicción por lote (batch).
    Acepta múltiples registros en una sola solicitud JSON.

    Ideal para:
    - Campañas de scoring masivo
    - Evaluación periódica de portafolio
    - Integración con procesos batch nocturnos
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    if len(request.records) == 0:
        raise HTTPException(status_code=400, detail="El lote no puede estar vacío.")

    if len(request.records) > 10_000:
        raise HTTPException(status_code=400,
                            detail="Máximo 10,000 registros por lote.")

    try:
        df_features = prepare_input(request.records)
        preds, probs = make_prediction(df_features, request.threshold)

        predictions = []
        for i in range(len(preds)):
            prob_default = float(probs[i][0])
            prob_ontime  = float(probs[i][1])
            predictions.append(PredictionResponse(
                prediction         = int(preds[i]),
                probability_default= round(prob_default, 4),
                probability_ontime = round(prob_ontime,  4),
                risk_label         = get_risk_label(prob_ontime),
                timestamp          = datetime.now().isoformat()
            ))

        n_default = int(sum(p.prediction == 0 for p in predictions))
        n_ontime  = int(sum(p.prediction == 1 for p in predictions))
        total     = len(predictions)

        log.info(f"Batch predicción: {total} registros | "
                 f"Incumplimiento: {n_default} ({n_default/total*100:.1f}%)")

        return BatchResponse(
            predictions     = predictions,
            total           = total,
            predicted_default = n_default,
            predicted_ontime  = n_ontime,
            default_rate    = round(n_default / total, 4)
        )

    except Exception as e:
        log.error(f"Error en batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DOCKERFILE (se imprime al ejecutar directamente)
# ═══════════════════════════════════════════════════════════════════════════════

DOCKERFILE_CONTENT = """FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc g++ libgomp1 && \\
    rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY mlops_pipeline/ ./mlops_pipeline/
COPY models/ ./models/

# Exponer puerto de la API
EXPOSE 8000

# Variable de entorno
ENV PYTHONPATH=/app/mlops_pipeline/src

# Comando de inicio
CMD ["uvicorn", "mlops_pipeline.src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
"""

DOCKERIGNORE_CONTENT = """__pycache__/
*.py[cod]
*.egg-info/
.git/
.env
venv/
env/
*.log
.ipynb_checkpoints/
"""


def save_docker_files():
    """Guarda Dockerfile y .dockerignore en la raíz del proyecto."""
    dockerfile_path = os.path.join(BASE_DIR, "Dockerfile")
    dockerignore_path = os.path.join(BASE_DIR, ".dockerignore")

    with open(dockerfile_path, "w") as f:
        f.write(DOCKERFILE_CONTENT.strip())

    with open(dockerignore_path, "w") as f:
        f.write(DOCKERIGNORE_CONTENT.strip())

    log.info(f"Dockerfile guardado: {dockerfile_path}")
    log.info(f".dockerignore guardado: {dockerignore_path}")


if __name__ == "__main__":
    import uvicorn

    # Guardar Dockerfile
    save_docker_files()
    print("Dockerfile y .dockerignore creados.")
    print()
    print("Para iniciar el servidor:")
    print("  uvicorn model_deploy:app --host 0.0.0.0 --port 8000 --reload")
    print()
    print("Para construir la imagen Docker:")
    print("  docker build -t mlops-creditos:latest .")
    print("  docker run -p 8000:8000 mlops-creditos:latest")
    print()
    print("Documentación interactiva (Swagger):")
    print("  http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
