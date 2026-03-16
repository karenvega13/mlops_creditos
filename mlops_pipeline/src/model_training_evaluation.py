"""
model_training_evaluation.py
─────────────────────────────────────────────────────────────────
Entrenamiento y Evaluación de Modelos — Proyecto Créditos
Institución Financiera | Área de Datos y Analítica

Descripción:
    Entrena múltiples modelos de clasificación supervisados,
    evalúa su desempeño con métricas apropiadas para datasets
    desbalanceados y selecciona el mejor modelo para producción.

Modelos:
    - Logistic Regression (baseline)
    - Random Forest
    - XGBoost
    - LightGBM

Métricas: ROC-AUC, F1, Precision, Recall, PR-AUC
─────────────────────────────────────────────────────────────────
"""

import os
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, f1_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from ft_engineering import run_feature_engineering, MODELS_DIR

warnings.filterwarnings("ignore")

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ─── Directorio de resultados ────────────────────────────────────────────────
RESULTS_DIR = os.path.join(MODELS_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS     = 5


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FUNCIONES REUTILIZABLES
# ═══════════════════════════════════════════════════════════════════════════════

def summarize_classification(y_true, y_pred, y_prob, model_name: str) -> dict:
    """
    Genera un resumen completo de métricas de clasificación.
    Función reutilizable para cualquier modelo binario.

    Args:
        y_true     : Etiquetas reales.
        y_pred     : Etiquetas predichas (umbral 0.5 por defecto).
        y_prob     : Probabilidades de la clase positiva.
        model_name : Nombre del modelo para el reporte.

    Returns:
        dict con todas las métricas clave.
    """
    roc_auc  = roc_auc_score(y_true, y_prob)
    pr_auc   = average_precision_score(y_true, y_prob)
    f1       = f1_score(y_true, y_pred, zero_division=0)
    prec     = precision_score(y_true, y_pred, zero_division=0)
    rec      = recall_score(y_true, y_pred, zero_division=0)
    cm       = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    summary = {
        "model"        : model_name,
        "roc_auc"      : round(roc_auc, 4),
        "pr_auc"       : round(pr_auc, 4),
        "f1_score"     : round(f1, 4),
        "precision"    : round(prec, 4),
        "recall"       : round(rec, 4),
        "specificity"  : round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0,
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn)
    }

    log.info(f"\n{'─'*55}")
    log.info(f"  MODELO: {model_name}")
    log.info(f"  ROC-AUC   : {roc_auc:.4f}")
    log.info(f"  PR-AUC    : {pr_auc:.4f}")
    log.info(f"  F1-Score  : {f1:.4f}")
    log.info(f"  Precision : {prec:.4f}")
    log.info(f"  Recall    : {rec:.4f}")
    log.info(f"{'─'*55}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['No Pagó (0)', 'Pagó (1)'])}")

    return summary


def build_model(model_name: str, class_weight: str = "balanced") -> object:
    """
    Fábrica de modelos. Retorna una instancia del clasificador
    configurado con hiperparámetros iniciales razonables.

    Args:
        model_name   : Nombre del modelo ('lr', 'rf', 'xgb', 'lgbm').
        class_weight : Manejo del desbalance de clases.

    Returns:
        Instancia del clasificador sklearn-compatible.
    """
    models = {
        "lr": LogisticRegression(
            C=0.1,
            class_weight=class_weight,
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE
        ),
        "rf": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "xgb": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=20,   # ~ratio de desbalance (511/10252 ≈ 20x)
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "lgbm": LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            class_weight=class_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
    }

    if model_name not in models:
        raise ValueError(f"Modelo '{model_name}' no disponible. Opciones: {list(models.keys())}")

    return models[model_name]


def plot_confusion_matrix(cm: np.ndarray, model_name: str, ax=None):
    """Visualiza la matriz de confusión."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Pagó (0)", "Pagó (1)"],
                yticklabels=["No Pagó (0)", "Pagó (1)"])
    ax.set_title(f"Matriz de Confusión\n{model_name}", fontweight="bold")
    ax.set_ylabel("Real")
    ax.set_xlabel("Predicho")


def plot_roc_curves(results_list: list, y_test):
    """
    Grafica curvas ROC comparativas para todos los modelos.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ROC Curves
    for res in results_list:
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        axes[0].plot(fpr, tpr, lw=2, label=f"{res['model']} (AUC={res['metrics']['roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title("Curvas ROC — Comparación de Modelos", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curves
    for res in results_list:
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, res["y_prob"])
        axes[1].plot(rec_arr, prec_arr, lw=2,
                     label=f"{res['model']} (PR-AUC={res['metrics']['pr_auc']:.3f})")
    axes[1].set_title("Curvas Precision-Recall — Comparación", fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_pr_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()
    log.info("Gráfico ROC/PR guardado.")


def plot_metrics_comparison(summary_df: pd.DataFrame):
    """
    Gráfico de barras comparativo de métricas principales.
    """
    metrics_to_plot = ["roc_auc", "pr_auc", "f1_score", "precision", "recall"]
    df_plot = summary_df.set_index("model")[metrics_to_plot]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(df_plot))
    width = 0.15

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]
    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        bars = ax.bar(x + i * width, df_plot[metric], width, label=metric.upper(),
                      color=color, alpha=0.85, edgecolor="black")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5)

    ax.set_xlabel("Modelo", fontweight="bold")
    ax.set_ylabel("Valor de la Métrica", fontweight="bold")
    ax.set_title("Comparación de Métricas por Modelo", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df_plot.index, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()
    log.info("Gráfico comparativo de métricas guardado.")


def plot_feature_importance(model, feature_names: list, model_name: str, top_n: int = 20):
    """Grafica importancia de features para modelos basados en árboles."""
    if not hasattr(model, "feature_importances_"):
        log.info(f"{model_name} no tiene feature_importances_.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    names   = [feature_names[i] for i in indices]
    vals    = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), vals[::-1], color="#2196F3", edgecolor="black", alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names[::-1])
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontweight="bold")
    ax.set_xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"feat_importance_{model_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ENTRENAMIENTO Y EVALUACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    feature_names: list,
    use_smote: bool = True
):
    """
    Entrena todos los modelos, aplica SMOTE si se requiere,
    evalúa en el conjunto de test y genera reportes completos.

    Args:
        X_train, X_test : Arrays de features (ya transformados).
        y_train, y_test : Series de etiquetas.
        feature_names   : Nombres de los features.
        use_smote       : Si True aplica SMOTE en entrenamiento.

    Returns:
        summary_df   (DataFrame con métricas de todos los modelos)
        best_model   (clasificador con mejor ROC-AUC)
    """
    # Aplicar SMOTE solo al conjunto de entrenamiento
    if use_smote:
        log.info("Aplicando SMOTE para balanceo de clases...")
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        log.info(f"Distribución post-SMOTE: {dict(pd.Series(y_train_bal).value_counts())}")
    else:
        X_train_bal, y_train_bal = X_train, y_train

    model_configs = [
        ("Logistic Regression", "lr"),
        ("Random Forest",       "rf"),
        ("XGBoost",             "xgb"),
        ("LightGBM",            "lgbm"),
    ]

    all_results  = []
    summary_list = []

    for model_name, model_key in model_configs:
        log.info(f"\nEntrenando: {model_name}...")
        model = build_model(model_key)
        model.fit(X_train_bal, y_train_bal)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = summarize_classification(y_test, y_pred, y_prob, model_name)
        summary_list.append(metrics)

        all_results.append({
            "model"  : model_name,
            "metrics": metrics,
            "y_pred" : y_pred,
            "y_prob" : y_prob,
            "clf"    : model
        })

        # Guardar modelo
        model_path = os.path.join(MODELS_DIR, f"model_{model_key}.pkl")
        joblib.dump(model, model_path)
        log.info(f"Modelo guardado: {model_path}")

        # Feature importance
        plot_feature_importance(model, feature_names, model_name)

    # ── Gráficos comparativos ──────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_list)
    plot_roc_curves(all_results, y_test)
    plot_metrics_comparison(summary_df)

    # ── Matrices de confusión ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for i, res in enumerate(all_results):
        cm = confusion_matrix(y_test, res["y_pred"])
        plot_confusion_matrix(cm, res["model"], ax=axes[i])
    plt.suptitle("Matrices de Confusión — Todos los Modelos", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # ── Tabla resumen ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TABLA RESUMEN DE EVALUACIÓN DE MODELOS")
    print("=" * 80)
    display_cols = ["model", "roc_auc", "pr_auc", "f1_score", "precision", "recall", "specificity"]
    print(summary_df[display_cols].to_string(index=False))
    print("=" * 80)

    # ── Mejor modelo ──────────────────────────────────────────────────────
    best_idx   = summary_df["roc_auc"].idxmax()
    best_name  = summary_df.loc[best_idx, "model"]
    best_model = [r["clf"] for r in all_results if r["model"] == best_name][0]

    log.info(f"\n🏆 MEJOR MODELO: {best_name}")
    log.info(f"   ROC-AUC  : {summary_df.loc[best_idx, 'roc_auc']}")
    log.info(f"   F1-Score : {summary_df.loc[best_idx, 'f1_score']}")

    # Guardar mejor modelo como "best_model.pkl"
    best_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, best_path)
    log.info(f"Mejor modelo guardado en: {best_path}")

    # Guardar tabla resumen
    summary_df.to_csv(os.path.join(RESULTS_DIR, "model_summary.csv"), index=False)

    return summary_df, best_model


# ═══════════════════════════════════════════════════════════════════════════════
# 3. VALIDACIÓN CRUZADA DEL MEJOR MODELO
# ═══════════════════════════════════════════════════════════════════════════════

def cross_validate_best(X_train, y_train, model, model_name: str):
    """Ejecuta validación cruzada estratificada del mejor modelo."""
    log.info(f"\nValidación cruzada ({CV_FOLDS}-fold) — {model_name}...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_results = cross_validate(
        model, X_train, y_train,
        cv=skf,
        scoring=["roc_auc", "f1", "average_precision"],
        return_train_score=True,
        n_jobs=-1
    )

    print(f"\n{'─'*55}")
    print(f"VALIDACIÓN CRUZADA — {model_name}")
    print(f"{'─'*55}")
    for metric in ["roc_auc", "f1", "average_precision"]:
        test_scores = cv_results[f"test_{metric}"]
        print(f"  {metric:25s}: {test_scores.mean():.4f} ± {test_scores.std():.4f}")
    print(f"{'─'*55}")

    return cv_results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EJECUCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
    log.info("=" * 60)

    # Feature Engineering
    X_train, X_test, y_train, y_test, preprocessor, feature_names = \
        run_feature_engineering()

    # Entrenamiento y evaluación
    summary_df, best_model = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        feature_names,
        use_smote=True
    )

    # Validación cruzada del mejor
    best_name = summary_df.loc[summary_df["roc_auc"].idxmax(), "model"]
    cross_validate_best(X_train, y_train, best_model, best_name)

    print("\n✅ Pipeline de entrenamiento completado.")
    print(f"   Modelos guardados en: {MODELS_DIR}")
    print(f"   Resultados en: {RESULTS_DIR}")
