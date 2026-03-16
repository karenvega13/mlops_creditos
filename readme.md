# 🏦 MLOps Pipeline — Modelo Predictivo de Créditos

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)](https://streamlit.io)

---

## 📋 Caso de Negocio

Una institución financiera colombiana requiere un **modelo predictivo de machine learning** que, usando información histórica de créditos, anticipe si un nuevo cliente **pagará su crédito a tiempo**.

El modelo apoya la toma de decisiones crediticias, permitiendo:
- Segmentar clientes por nivel de riesgo antes del desembolso.
- Optimizar el proceso de aprobación de créditos.
- Reducir la tasa de cartera vencida.

**Variable objetivo:** `Pago_atiempo` — 1 = Pagó a tiempo, 0 = No pagó.

---

## 🗂️ Estructura del Proyecto

```
mlops_pipeline/
│
└── src/
    ├── Cargar_datos.ipynb              # Carga del dataset (no productivo)
    ├── comprension_eda.ipynb           # Análisis exploratorio de datos (EDA)
    ├── ft_engineering.py               # Pipeline de Feature Engineering
    ├── model_training_evaluation.py    # Entrenamiento y evaluación de modelos
    ├── model_monitoring.py             # Monitoreo y detección de data drift
    ├── model_deploy.py                 # Despliegue como API REST (FastAPI)
    └── app_monitoring.py               # Dashboard Streamlit de monitoreo
│
Base_de_datos.csv                       # Dataset histórico de créditos
requirements.txt                        # Dependencias del proyecto
.gitignore                              # Archivos excluidos del control de versiones
readme.md                               # Este archivo
Dockerfile                              # Imagen Docker para producción
```

---

## 🔬 Dataset

| Propiedad | Valor |
|---|---|
| Registros | 10,763 |
| Variables | 23 |
| Variable objetivo | `Pago_atiempo` (binaria) |
| Distribución clases | 95.2% paga a tiempo / 4.8% no paga |
| Rango de fechas | Nov 2024 — Abr 2026 |

**⚠️ Desbalance de clases:** El dataset está altamente desbalanceado (~20:1). Se utiliza SMOTE para balanceo en entrenamiento y métricas ajustadas (ROC-AUC, F1, PR-AUC).

---

## 🏗️ Pipeline de MLOps

### Avance 1 — Versionamiento y EDA
- ✅ Repositorio GitHub con estructura de carpetas definida
- ✅ Tres ramas: `developer`, `certification`, `main`
- ✅ `Cargar_datos.ipynb`: Carga, validación y diccionario de datos
- ✅ `comprension_eda.ipynb`: EDA completo con análisis uni/bi/multivariable

### Avance 2 — Feature Engineering y Modelado
- ✅ `ft_engineering.py`: Pipeline con ColumnTransformer
  - Numéricas: SimpleImputer → StandardScaler
  - Categóricas: SimpleImputer → OneHotEncoder
  - Ordinales: SimpleImputer → OrdinalEncoder
- ✅ `model_training_evaluation.py`: 4 modelos + SMOTE + evaluación comparativa

### Avance 3 — Monitoreo y Dashboard
- ✅ `model_monitoring.py`: KS Test, PSI, Jensen-Shannon, Chi-cuadrado
- ✅ Dashboard Streamlit con semáforo de alertas y análisis temporal

### Avance 4 — Despliegue
- ✅ `model_deploy.py`: API REST con FastAPI
  - `POST /predict` — predicción individual
  - `POST /predict/batch` — predicción por lotes
- ✅ Dockerfile para contenedorización

---

## 🤖 Modelos Entrenados

| Modelo | Descripción |
|---|---|
| Logistic Regression | Baseline lineal |
| Random Forest | Ensemble de árboles de decisión |
| **XGBoost** | Gradient Boosting optimizado (**mejor modelo**) |
| LightGBM | Gradient Boosting eficiente |

**Métricas de selección:** ROC-AUC (principal), F1-Score, PR-AUC, Recall.

---

## 🔑 Principales Hallazgos del EDA

1. **Desbalance crítico (95%/5%):** Requiere técnicas especializadas (SMOTE, `class_weight='balanced'`).
2. **`saldo_mora` es la variable más discriminante** — correlación negativa directa con pago a tiempo.
3. **`tendencia_ingresos` tiene 27% de nulos** y datos sucios (valores numéricos erróneos) — limpieza necesaria.
4. **Alta multicolinealidad** entre `saldo_total` y `saldo_principal` (r=0.96) — se elimina una.
5. **Outliers extremos** en `salario_cliente` (hasta $22,000 millones) y `capital_prestado` — capping P99.5.
6. **`puntaje_datacredito`** tiene poder predictivo: clientes con bajo score tienen mayor incumplimiento.
7. **Clientes independientes** tienen ligeramente mayor tasa de incumplimiento que empleados.
8. **`tendencia_ingresos = Decreciente`** → mayor tasa de incumplimiento (relación lógica confirmada).

---

## 📦 Instalación y Ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/karenvega13/mlops_creditos.git
cd mlops_creditos
```

### 2. Crear entorno virtual e instalar dependencias
```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. Ejecutar el pipeline completo
```bash
# Feature Engineering
python mlops_pipeline/src/ft_engineering.py

# Entrenamiento y evaluación
python mlops_pipeline/src/model_training_evaluation.py

# Monitoreo
python mlops_pipeline/src/model_monitoring.py
```

### 4. Iniciar la API REST
```bash
cd mlops_pipeline/src
uvicorn model_deploy:app --host 0.0.0.0 --port 8000 --reload
# Documentación: http://localhost:8000/docs
```

### 5. Iniciar el Dashboard de Monitoreo
```bash
streamlit run mlops_pipeline/src/app_monitoring.py
```

### 6. Docker
```bash
docker build -t mlops-creditos:latest .
docker run -p 8000:8000 mlops-creditos:latest
```

---

## 🌿 Estrategia de Ramas y Versiones

| Versión | Rama | Contenido |
|---|---|---|
| V1.0.0 | `developer`, `certification`, `main` | Estructura inicial de carpetas |
| V1.0.1 | `developer` → `main` | Notebooks EDA (Cargar_datos + EDA) |
| V1.1.0 | `developer` → `main` | Feature Engineering + Modelado |
| V1.1.1 | `developer` → `main` | Monitoreo + Deploy + README |

---

## 🔄 Reglas de Validación de Datos

| Variable | Regla |
|---|---|
| `edad_cliente` | 18–90 años |
| `salario_cliente` | ≥ 0 |
| `capital_prestado` | > 0 |
| `plazo_meses` | 1–120 meses |
| `tendencia_ingresos` | Estable / Creciente / Decreciente / NaN |
| `tipo_laboral` | Empleado / Independiente |
| `puntaje_datacredito` | -10–999 |
| `Pago_atiempo` | 0 o 1 |

---

## 👩‍💻 Autor

**Angélica** — Científico de Datos Junior Advanced
Módulo: Fundamentos de Nube y Ciencia de Datos de Producción
Email: karenvega081@gmail.com
