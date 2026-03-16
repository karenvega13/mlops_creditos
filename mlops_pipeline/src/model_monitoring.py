"""
model_monitoring.py
─────────────────────────────────────────────────────────────────
Monitoreo del Modelo y Detección de Data Drift — Proyecto Créditos
Institución Financiera | Área de Datos y Analítica

Descripción:
    Trabajo de monitoreo que obtiene datos con predicciones
    y calcula métricas estadísticas para detectar cambios
    en la población que puedan afectar el desempeño del modelo.

Métricas de Drift:
    - Kolmogorov-Smirnov (KS) para numéricas
    - Population Stability Index (PSI)
    - Jensen-Shannon Divergence
    - Chi-Cuadrado para categóricas

Interfaz: Streamlit (app visual con semáforo de alertas)
─────────────────────────────────────────────────────────────────
"""

import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime, timedelta
from scipy import stats
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ─── Rutas ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DATA_PATH   = os.path.join(BASE_DIR, "Base_de_datos.csv")
MONITOR_DIR = os.path.join(BASE_DIR, "monitoring")
os.makedirs(MONITOR_DIR, exist_ok=True)

# ─── Umbrales de alerta ──────────────────────────────────────────────────────
THRESHOLDS = {
    "ks_statistic"     : {"green": 0.10, "yellow": 0.20},   # < verde, < amarillo → rojo
    "psi"              : {"green": 0.10, "yellow": 0.20},
    "js_divergence"    : {"green": 0.10, "yellow": 0.20},
    "chi2_pvalue"      : {"green": 0.05, "yellow": 0.01},   # > verde OK (invertido)
}

# ─── Variables a monitorear ──────────────────────────────────────────────────
NUMERIC_MONITOR = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "cuota_pactada", "puntaje", "puntaje_datacredito", "saldo_mora",
    "cant_creditosvigentes", "huella_consulta"
]
CATEG_MONITOR = ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FUNCIONES DE CÁLCULO DE DRIFT
# ═══════════════════════════════════════════════════════════════════════════════

def ks_test(reference: np.ndarray, current: np.ndarray) -> dict:
    """
    Kolmogorov-Smirnov test para variables numéricas.
    H0: Las dos muestras provienen de la misma distribución.
    """
    stat, pvalue = stats.ks_2samp(reference, current)
    return {"ks_statistic": round(stat, 4), "ks_pvalue": round(pvalue, 4)}


def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.10: Sin cambio significativo (verde)
    PSI 0.10–0.20: Cambio moderado (amarillo)
    PSI > 0.20: Cambio significativo — reentrenar (rojo)
    """
    # Crear bins basados en referencia
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current,   bins=bin_edges)

    # Evitar divisiones por cero
    ref_pct = np.where(ref_counts == 0, 0.0001, ref_counts / len(reference))
    cur_pct = np.where(cur_counts == 0, 0.0001, cur_counts / len(current))

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return round(float(psi), 4)


def jensen_shannon_div(reference: np.ndarray, current: np.ndarray, bins: int = 20) -> float:
    """
    Jensen-Shannon Divergence — versión simétrica de KL-divergence.
    Rango [0, 1]: 0 = distribuciones idénticas, 1 = totalmente distintas.
    """
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    p, _ = np.histogram(reference, bins=bin_edges, density=True)
    q, _ = np.histogram(current,   bins=bin_edges, density=True)

    p = p + 1e-10
    q = q + 1e-10

    return round(float(jensenshannon(p, q)), 4)


def chi2_test_categorical(reference: pd.Series, current: pd.Series) -> dict:
    """
    Chi-cuadrado para variables categóricas.
    Compara la distribución de frecuencias entre referencia y actual.
    """
    all_cats = set(reference.unique()) | set(current.unique())
    ref_counts = reference.value_counts().reindex(all_cats, fill_value=0)
    cur_counts = current.value_counts().reindex(all_cats, fill_value=0)

    chi2, pvalue = stats.chisquare(f_obs=cur_counts, f_exp=ref_counts * len(current) / len(reference))
    return {"chi2_statistic": round(chi2, 4), "chi2_pvalue": round(pvalue, 6)}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CLASIFICACIÓN DE ALERTAS
# ═══════════════════════════════════════════════════════════════════════════════

def classify_alert(metric_name: str, value: float) -> str:
    """
    Clasifica el nivel de alerta basado en umbrales predefinidos.
    Returns: 'green', 'yellow', o 'red'
    """
    th = THRESHOLDS.get(metric_name)
    if th is None:
        return "green"

    if metric_name == "chi2_pvalue":
        # Para p-valor: verde si > 0.05, amarillo si 0.01-0.05, rojo si < 0.01
        if value > th["green"]:
            return "green"
        elif value > th["yellow"]:
            return "yellow"
        else:
            return "red"
    else:
        if value < th["green"]:
            return "green"
        elif value < th["yellow"]:
            return "yellow"
        else:
            return "red"


def get_alert_message(variable: str, metrics: dict) -> str:
    """Genera mensaje de recomendación según nivel de alerta."""
    alerts = [v for v in metrics.values() if isinstance(v, str) and v == "red"]

    if alerts:
        return (f"⚠️  ALERTA ROJA en '{variable}': "
                f"Se detectó drift significativo. "
                f"Se recomienda reentrenar el modelo con datos recientes.")
    amber = [v for v in metrics.values() if isinstance(v, str) and v == "yellow"]
    if amber:
        return (f"🟡 ALERTA AMARILLA en '{variable}': "
                f"Cambio moderado detectado. Monitorear de cerca.")
    return f"✅ '{variable}': Sin drift significativo."


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MUESTREO PERIÓDICO Y CÁLCULO DE MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_monitoring_sample(df: pd.DataFrame, reference_df: pd.DataFrame,
                                period: str = "month") -> dict:
    """
    Toma una muestra de los datos actuales y calcula métricas de drift
    comparando con el conjunto de referencia (datos de entrenamiento).

    Args:
        df           : DataFrame actual (nuevos datos con predicciones).
        reference_df : DataFrame de referencia (datos de entrenamiento).
        period       : Granularidad del análisis ('month', 'week').

    Returns:
        dict con métricas de drift por variable.
    """
    drift_report = {
        "timestamp"  : datetime.now().isoformat(),
        "period"     : period,
        "n_reference": len(reference_df),
        "n_current"  : len(df),
        "variables"  : {}
    }

    # ── Variables numéricas ────────────────────────────────────────────────
    for col in NUMERIC_MONITOR:
        if col not in df.columns or col not in reference_df.columns:
            continue

        ref = reference_df[col].dropna().values
        cur = df[col].dropna().values

        if len(cur) < 30:
            continue

        ks     = ks_test(ref, cur)
        psi    = calculate_psi(ref, cur)
        js     = jensen_shannon_div(ref, cur)

        # Alertas
        ks_alert  = classify_alert("ks_statistic",  ks["ks_statistic"])
        psi_alert = classify_alert("psi",            psi)
        js_alert  = classify_alert("js_divergence",  js)

        drift_report["variables"][col] = {
            "type"         : "numeric",
            "ks_statistic" : ks["ks_statistic"],
            "ks_pvalue"    : ks["ks_pvalue"],
            "ks_alert"     : ks_alert,
            "psi"          : psi,
            "psi_alert"    : psi_alert,
            "js_divergence": js,
            "js_alert"     : js_alert,
            "overall_alert": max([ks_alert, psi_alert, js_alert],
                                  key=lambda x: ["green", "yellow", "red"].index(x)),
            "ref_mean"     : round(ref.mean(), 4),
            "cur_mean"     : round(cur.mean(), 4),
            "ref_std"      : round(ref.std(), 4),
            "cur_std"      : round(cur.std(), 4),
            "message"      : get_alert_message(col, {
                "ks": ks_alert, "psi": psi_alert, "js": js_alert})
        }

    # ── Variables categóricas ──────────────────────────────────────────────
    for col in CATEG_MONITOR:
        if col not in df.columns or col not in reference_df.columns:
            continue

        ref_s = reference_df[col].dropna().astype(str)
        cur_s = df[col].dropna().astype(str)

        if len(cur_s) < 30:
            continue

        chi2_res = chi2_test_categorical(ref_s, cur_s)
        chi2_alert = classify_alert("chi2_pvalue", chi2_res["chi2_pvalue"])

        drift_report["variables"][col] = {
            "type"           : "categorical",
            "chi2_statistic" : chi2_res["chi2_statistic"],
            "chi2_pvalue"    : chi2_res["chi2_pvalue"],
            "chi2_alert"     : chi2_alert,
            "overall_alert"  : chi2_alert,
            "ref_dist"       : ref_s.value_counts(normalize=True).round(4).to_dict(),
            "cur_dist"       : cur_s.value_counts(normalize=True).round(4).to_dict(),
            "message"        : get_alert_message(col, {"chi2": chi2_alert})
        }

    # Guardar reporte
    report_path = os.path.join(MONITOR_DIR, f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w") as f:
        json.dump(drift_report, f, indent=2, default=str)
    log.info(f"Reporte de drift guardado: {report_path}")

    return drift_report


def generate_predictions_table(df: pd.DataFrame, model_path: str,
                                 preprocessor_path: str) -> pd.DataFrame:
    """
    Aplica el modelo al dataset actual y retorna la tabla con predicciones.
    Esta tabla es la que alimenta el monitoreo.
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from ft_engineering import clean_data, add_derived_features, NUMERIC_COLS, \
                               CATEG_NOMINAL_COLS, CATEG_ORDINAL_COLS

    model       = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    df_clean    = clean_data(df.copy())
    df_feat     = add_derived_features(df_clean)

    feature_cols = [c for c in NUMERIC_COLS + CATEG_NOMINAL_COLS + CATEG_ORDINAL_COLS
                    if c in df_feat.columns]

    X = preprocessor.transform(df_feat[feature_cols])
    df.loc[:, "score_predicho"] = model.predict_proba(X)[:, 1]
    df.loc[:, "prediccion"]     = model.predict(X)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. APLICACIÓN STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════════

STREAMLIT_APP = '''
"""
Aplicación Streamlit — Dashboard de Monitoreo del Modelo
Proyecto Créditos | Institución Financiera
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime

# ── Configuración de página ────────────────────────────────────────────────
st.set_page_config(
    page_title="🔍 Monitoreo del Modelo Crediticio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Rutas ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MONITOR_DIR = os.path.join(BASE_DIR, "monitoring")
DATA_PATH   = os.path.join(BASE_DIR, "Base_de_datos.csv")
sys.path.insert(0, os.path.dirname(__file__))

# ─── Paleta de colores de semáforo ────────────────────────────────────────
ALERT_COLORS = {"green": "#4CAF50", "yellow": "#FF9800", "red": "#F44336"}
ALERT_ICONS  = {"green": "✅", "yellow": "🟡", "red": "🔴"}
ALERT_LABELS = {"green": "Sin drift", "yellow": "Drift moderado", "red": "Drift significativo"}


# ─── Cargar datos ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["fecha_prestamo"])
    # Simular score de predicción para demostración
    np.random.seed(42)
    df["score_predicho"] = np.random.beta(8, 1, len(df))
    df["prediccion"]     = (df["score_predicho"] > 0.5).astype(int)
    df["periodo"]        = df["fecha_prestamo"].dt.to_period("M").astype(str)
    return df

@st.cache_data
def load_drift_reports():
    reports = []
    for f in sorted(glob.glob(os.path.join(MONITOR_DIR, "drift_report_*.json"))):
        with open(f) as fp:
            reports.append(json.load(fp))
    return reports


# ─── Calcular drift en tiempo real ────────────────────────────────────────
def compute_drift_live(df, ref_period, cur_period):
    from model_monitoring import generate_monitoring_sample
    ref_df = df[df["periodo"] == ref_period]
    cur_df = df[df["periodo"] == cur_period]
    if len(ref_df) < 30 or len(cur_df) < 30:
        return None
    return generate_monitoring_sample(cur_df, ref_df)


# ════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

st.title("📊 Dashboard de Monitoreo — Modelo Crediticio")
st.caption("Detección de Data Drift | Institución Financiera")

df = load_data()
periodos = sorted(df["periodo"].unique())

# ─── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuración")
ref_period = st.sidebar.selectbox("Período de Referencia",  periodos[:-1], index=0)
cur_period = st.sidebar.selectbox("Período Actual (actual)", periodos[1:],  index=len(periodos)-2)
var_sel    = st.sidebar.multiselect(
    "Variables a monitorear",
    options=["capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
             "cuota_pactada", "puntaje", "puntaje_datacredito", "saldo_mora",
             "cant_creditosvigentes", "tipo_credito", "tipo_laboral"],
    default=["capital_prestado", "puntaje_datacredito", "saldo_mora", "tipo_laboral"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Umbrales de alerta:**")
st.sidebar.markdown("🟢 PSI < 0.10 — Sin drift")
st.sidebar.markdown("🟡 PSI 0.10–0.20 — Moderado")
st.sidebar.markdown("🔴 PSI > 0.20 — Reentrenar")

# ─── Métricas globales ─────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
ref_df = df[df["periodo"] == ref_period]
cur_df = df[df["periodo"] == cur_period]

col1.metric("Registros Referencia", f"{len(ref_df):,}")
col2.metric("Registros Actuales",   f"{len(cur_df):,}",
            delta=f"{len(cur_df)-len(ref_df):+,}")
col3.metric("Tasa Incumplimiento Ref.",
            f"{(ref_df['Pago_atiempo']==0).mean()*100:.1f}%")
col4.metric("Tasa Incumplimiento Act.",
            f"{(cur_df['Pago_atiempo']==0).mean()*100:.1f}%",
            delta=f"{((cur_df['Pago_atiempo']==0).mean()-(ref_df['Pago_atiempo']==0).mean())*100:+.2f}%")

st.markdown("---")

# ─── Calcular drift ────────────────────────────────────────────────────────
drift_result = compute_drift_live(df, ref_period, cur_period)

# ─── Tabla de métricas de drift ────────────────────────────────────────────
st.subheader("📋 Tabla de Métricas de Drift por Variable")

if drift_result and drift_result.get("variables"):
    rows = []
    for var, m in drift_result["variables"].items():
        if var not in var_sel:
            continue
        if m["type"] == "numeric":
            rows.append({
                "Variable": var,
                "Tipo": "Numérica",
                "KS": m.get("ks_statistic", "—"),
                "PSI": m.get("psi", "—"),
                "JS Div.": m.get("js_divergence", "—"),
                "Media Ref.": m.get("ref_mean", "—"),
                "Media Act.": m.get("cur_mean", "—"),
                "Alerta": ALERT_ICONS.get(m["overall_alert"], "—") + " " + ALERT_LABELS.get(m["overall_alert"], "—"),
            })
        else:
            rows.append({
                "Variable": var,
                "Tipo": "Categórica",
                "KS": "—",
                "PSI": "—",
                "JS Div.": "—",
                "χ² p-val": m.get("chi2_pvalue", "—"),
                "Media Ref.": "—",
                "Media Act.": "—",
                "Alerta": ALERT_ICONS.get(m["overall_alert"], "—") + " " + ALERT_LABELS.get(m["overall_alert"], "—"),
            })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Seleccione variables en el sidebar para ver las métricas.")

st.markdown("---")

# ─── Gráficos comparativos de distribución ────────────────────────────────
st.subheader("📈 Comparación de Distribuciones: Referencia vs Actual")

num_vars = [v for v in var_sel if v in ref_df.select_dtypes(include="number").columns]
cat_vars = [v for v in var_sel if v in ref_df.select_dtypes(exclude="number").columns]

if num_vars:
    n_cols = min(3, len(num_vars))
    n_rows = (len(num_vars) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = [axes] if n_rows * n_cols == 1 else axes.flatten()

    for i, col in enumerate(num_vars):
        p1  = min(ref_df[col].quantile(0.01), cur_df[col].quantile(0.01))
        p99 = max(ref_df[col].quantile(0.99), cur_df[col].quantile(0.99))
        axes[i].hist(ref_df[col].clip(p1, p99).dropna(), bins=30,
                     alpha=0.5, color="#2196F3", label=f"Ref. ({ref_period})", density=True)
        axes[i].hist(cur_df[col].clip(p1, p99).dropna(), bins=30,
                     alpha=0.5, color="#F44336", label=f"Act. ({cur_period})", density=True)
        axes[i].set_title(col, fontweight="bold")
        axes[i].legend(fontsize=8)

    for j in range(len(num_vars), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)

if cat_vars:
    for col in cat_vars:
        fig, ax = plt.subplots(figsize=(9, 4))
        ref_pct = ref_df[col].value_counts(normalize=True).rename("Referencia")
        cur_pct = cur_df[col].value_counts(normalize=True).rename("Actual")
        comp = pd.concat([ref_pct, cur_pct], axis=1).fillna(0)
        comp.plot(kind="bar", ax=ax, color=["#2196F3", "#F44336"], alpha=0.8, edgecolor="black")
        ax.set_title(f"Distribución de '{col}' — Ref. vs Actual", fontweight="bold")
        ax.set_ylabel("Proporción")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

st.markdown("---")

# ─── Evolución temporal del drift ─────────────────────────────────────────
st.subheader("📅 Evolución Temporal del Drift")

numeric_key = "saldo_mora" if "saldo_mora" in var_sel else (num_vars[0] if num_vars else None)
if numeric_key and len(periodos) >= 3:
    from model_monitoring import calculate_psi
    psi_values, period_labels = [], []
    ref_arr = df[df["periodo"] == periodos[0]][numeric_key].dropna().values
    for p in periodos[1:]:
        cur_arr = df[df["periodo"] == p][numeric_key].dropna().values
        if len(cur_arr) >= 30:
            psi_values.append(calculate_psi(ref_arr, cur_arr))
            period_labels.append(p)

    if psi_values:
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = [ALERT_COLORS["green"] if v < 0.10 else
                  ALERT_COLORS["yellow"] if v < 0.20 else
                  ALERT_COLORS["red"] for v in psi_values]
        ax.bar(period_labels, psi_values, color=colors, edgecolor="black", alpha=0.85)
        ax.axhline(0.10, color="orange",  linestyle="--", label="Umbral amarillo (0.10)")
        ax.axhline(0.20, color="red",     linestyle="--", label="Umbral rojo (0.20)")
        ax.set_title(f"PSI a lo largo del tiempo — {numeric_key}", fontweight="bold")
        ax.set_ylabel("PSI")
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

st.markdown("---")

# ─── Mensajes de recomendación ────────────────────────────────────────────
st.subheader("💬 Recomendaciones Automáticas")

if drift_result and drift_result.get("variables"):
    red_vars    = [v for v, m in drift_result["variables"].items() if m["overall_alert"] == "red"]
    yellow_vars = [v for v, m in drift_result["variables"].items() if m["overall_alert"] == "yellow"]

    if red_vars:
        st.error(f"🔴 **ALERTA CRÍTICA:** Drift significativo en: {', '.join(red_vars)}. "
                 f"**Se recomienda reentrenar el modelo inmediatamente.**")
    if yellow_vars:
        st.warning(f"🟡 **ATENCIÓN:** Drift moderado detectado en: {', '.join(yellow_vars)}. "
                   f"Monitorear de cerca y evaluar si se requiere reentrenamiento.")
    if not red_vars and not yellow_vars:
        st.success("✅ **Todas las variables dentro de umbrales normales.** El modelo es estable.")

    # Mostrar mensajes individuales
    with st.expander("Ver mensajes detallados por variable"):
        for var, m in drift_result["variables"].items():
            if var in var_sel:
                st.markdown(f"**{var}:** {m.get('message', '')}")

st.markdown("---")
st.caption(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
           f"Proyecto Créditos — Institución Financiera")
'''


def save_streamlit_app():
    """Guarda la aplicación Streamlit como archivo independiente."""
    app_path = os.path.join(BASE_DIR, "mlops_pipeline", "src", "app_monitoring.py")
    with open(app_path, "w", encoding="utf-8") as f:
        # Extraer solo el código (sin las comillas triples)
        code = STREAMLIT_APP.strip().lstrip('"\n').rstrip('"')
        f.write(code)
    log.info(f"App Streamlit guardada en: {app_path}")
    return app_path


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EJECUCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("MONITOREO DEL MODELO — DATA DRIFT")
    log.info("=" * 60)

    # Cargar datos
    df = pd.read_csv(DATA_PATH, parse_dates=["fecha_prestamo"])

    # Crear periodos mensuales
    df["periodo"] = df["fecha_prestamo"].dt.to_period("M").astype(str)
    periodos = sorted(df["periodo"].unique())

    log.info(f"Periodos disponibles: {periodos}")

    # Calcular drift entre primer y último período
    if len(periodos) >= 2:
        ref_df = df[df["periodo"] == periodos[0]]
        cur_df = df[df["periodo"] == periodos[-1]]

        log.info(f"Referencia: {periodos[0]} ({len(ref_df)} registros)")
        log.info(f"Actual    : {periodos[-1]} ({len(cur_df)} registros)")

        report = generate_monitoring_sample(cur_df, ref_df, period="month")

        # Mostrar resumen
        print("\n" + "=" * 60)
        print("RESUMEN DE DATA DRIFT")
        print("=" * 60)
        for var, metrics in report["variables"].items():
            alert = metrics["overall_alert"].upper()
            icon  = {"GREEN": "✅", "YELLOW": "🟡", "RED": "🔴"}.get(alert, "❓")
            print(f"  {icon} {var:35s} → {alert}")
        print("=" * 60)

    # Generar app de Streamlit
    app_path = save_streamlit_app()
    print(f"\n✅ Monitoreo configurado.")
    print(f"   Para ejecutar el dashboard:")
    print(f"   streamlit run {app_path}")
