import os
from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.config import DATASET_FORMATS
from utils.file_utils import ensure_dir


# Estilo global consistente con el resto de los módulos de evaluación
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})


class QrelsDifficultyAnalyzer:
    """
    Analiza la distribución de los juicios de relevancia (qrels) y
    su relación con la dificultad de las consultas medida por métricas
    como AP o nDCG.

    Español: Este módulo está pensado para generar figuras que permitan
    caracterizar la calidad de los qrels y la distribución de dificultad
    de las consultas, en línea con la discusión teórica de la memoria.
    """

    def __init__(
        self,
        qrels_df: pd.DataFrame,
        evaluation_results: Dict,
        dataset_name: str,
        output_dir: Optional[str] = None,
        difficulty_metrics: Optional[List[str]] = None,
    ):
        self.dataset_name = dataset_name
        self.output_dir = ensure_dir(output_dir) if output_dir else None
        # Métricas preferidas para definir dificultad (por ejemplo, ['ap', 'ndcg@10'])
        self.difficulty_metrics = difficulty_metrics or []
        # Estructuras para exportar datos consolidados de qrels y dificultad
        self.qrels_relevance_distribution = None
        self.qrels_relevant_docs_per_query = None
        self.difficulty_details: Dict[str, Dict] = {}

        dataset_config = DATASET_FORMATS.get(
            dataset_name,
            DATASET_FORMATS["antique_test"],
        )

        self.binary_threshold = dataset_config.get("binary_threshold", 1)
        self.relevance_levels = dataset_config.get("relevance_levels", {})

        # Normalizar nombres de columnas de qrels como en evaluator.py
        default_qrels_columns = {"qid": "query_id", "docno": "doc_id", "label": "relevance"}
        column_map = dataset_config.get("qrels_columns", default_qrels_columns)
        self.qrels = qrels_df.rename(
            columns={k: v for k, v in column_map.items() if k in qrels_df.columns}
        ).copy()

        # Asegurar tipos básicos
        if "query_id" in self.qrels.columns:
            self.qrels["query_id"] = self.qrels["query_id"].astype(str)
        if "relevance" in self.qrels.columns:
            self.qrels["relevance"] = self.qrels["relevance"].astype(int)

        # Extraer métricas por consulta de evaluation_results
        per_query_data = {}
        for metric, metric_data in evaluation_results.items():
            per_query = metric_data.get("per_query", {})
            if per_query:
                # Convertir llaves de consulta a string para alinear
                per_query_data[metric] = {str(qid): val for qid, val in per_query.items()}

        self.metrics_df = pd.DataFrame(per_query_data) if per_query_data else pd.DataFrame()

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    def _save_plot(self, filename: str):
        """
        Guardar gráficos en PNG y PDF, saneando el nombre para Windows.
        """
        if not self.output_dir:
            plt.show()
            return

        ensure_dir(self.output_dir)
        safe_base = filename.replace(" ", "_")
        base_path = os.path.normpath(os.path.join(self.output_dir, safe_base))

        plt.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{base_path}.pdf", bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Análisis de qrels
    # ------------------------------------------------------------------
    def plot_relevance_level_distribution(self):
        """
        Muestra la distribución de niveles de relevancia en los qrels.

        Español: Permite ver cuántos juicios hay en cada nivel (no relevante,
        marginal, altamente relevante, etc.), lo que da contexto sobre la
        calidad y granularidad de los qrels.
        """
        if "relevance" not in self.qrels.columns:
            return

        counts = self.qrels["relevance"].value_counts().sort_index()
        # Guardar distribución de niveles de relevancia para análisis posterior
        self.qrels_relevance_distribution = counts.to_dict()
        labels = [
            self.relevance_levels.get(level, str(level))
            for level in counts.index
        ]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=labels, y=counts.values, palette="viridis")
        plt.title(f"Distribución de niveles de relevancia (qrels) - {self.dataset_name}")
        plt.xlabel("Nivel de relevancia")
        plt.ylabel("Número de juicios")
        plt.xticks(rotation=30, ha="right")

        for i, v in enumerate(counts.values):
            plt.text(i, v, str(v), ha="center", va="bottom", fontsize=10)

        self._save_plot("qrels_distribucion_niveles_relevancia")


    # ------------------------------------------------------------------
    # Análisis de dificultad de consultas
    # ------------------------------------------------------------------
    def plot_difficulty_classes(
        self,
        metric: str = "ap",
        hard_percentile: float = 0.2,
        easy_percentile: float = 0.8,
    ):
        """
        Clasifica las consultas en fáciles/intermedias/difíciles según percentiles
        de una métrica de efectividad (por defecto AP).

        Español: Implementa una definición basada en percentiles como las descritas
        en la memoria (clases ordenadas de dificultad). El 20% inferior se marca
        como "difícil" y el 20% superior como "fácil" (valores por defecto).
        """
        if self.metrics_df.empty or metric not in self.metrics_df.columns:
            return

        scores = self.metrics_df[metric].dropna()
        if scores.empty:
            return

        hard_thr = scores.quantile(hard_percentile)
        easy_thr = scores.quantile(easy_percentile)

        def classify(val: float) -> str:
            if val <= hard_thr:
                return "Difícil (percentil bajo)"
            if val >= easy_thr:
                return "Fácil (percentil alto)"
            return "Intermedia"

        difficulty_labels = scores.apply(classify)
        counts = difficulty_labels.value_counts().reindex(
            ["Difícil (percentil bajo)", "Intermedia", "Fácil (percentil alto)"]
        ).fillna(0)

        plt.figure(figsize=(8, 6))
        metric_label = "AP" if metric == "ap" else metric
        sns.barplot(x=counts.index, y=counts.values, palette="rocket")
        plt.title(f"Distribución de dificultad de consultas según {metric_label}")
        plt.xlabel("Clase de dificultad")
        plt.ylabel("Número de consultas")
        plt.xticks(rotation=20, ha="right")

        for i, v in enumerate(counts.values):
            if pd.notna(v):
                plt.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=10)

        self._save_plot(f"dificultad_consultas_{metric}")

        # Guardar resumen numérico para ser citado en la memoria
        if self.output_dir:
            # Conversión robusta a enteros, manejando posibles NaN
            hard_count = counts.get("Difícil (percentil bajo)", 0)
            inter_count = counts.get("Intermedia", 0)
            easy_count = counts.get("Fácil (percentil alto)", 0)
            summary = {
                "metric": metric,
                "hard_percentile": hard_percentile,
                "easy_percentile": easy_percentile,
                "thresholds": {
                    "hard": float(hard_thr),
                    "easy": float(easy_thr),
                },
                "counts": {
                    "hard": int(hard_count) if not pd.isna(hard_count) else 0,
                    "intermediate": int(inter_count) if not pd.isna(inter_count) else 0,
                    "easy": int(easy_count) if not pd.isna(easy_count) else 0,
                },
            }
            import json

            summary_path = os.path.join(
                self.output_dir, f"dificultad_consultas_{metric}.json"
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            # Guardar detalles por consulta para este métrica (valor y clase)
            per_query_info = {}
            for qid, value in scores.items():
                label = difficulty_labels.get(qid, "Intermedia")
                per_query_info[str(qid)] = {
                    "score": float(value),
                    "class": label,
                }
            self.difficulty_details[metric] = {
                "summary": summary,
                "per_query": per_query_info,
            }


    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------
    def generate_all_plots(self):
        """
        Genera todas las figuras relacionadas con qrels y dificultad
        de consultas.
        """
        self.plot_relevance_level_distribution()

        # Seleccionar métricas para dificultad:
        # 1) las indicadas explícitamente (si existen en metrics_df),
        # 2) en caso contrario, todas las métricas disponibles.
        if not self.metrics_df.empty:
            if self.difficulty_metrics:
                metrics_to_use = [
                    m for m in self.difficulty_metrics if m in self.metrics_df.columns
                ]
            else:
                metrics_to_use = list(self.metrics_df.columns)
        else:
            metrics_to_use = []

        for metric in metrics_to_use:
            self.plot_difficulty_classes(metric)

        # Exportar un archivo JSON consolidado con qrels y dificultad
        if self.output_dir:
            import json

            export_data = {
                "dataset": self.dataset_name,
                "qrels": {
                    "binary_threshold": self.binary_threshold,
                    "relevance_distribution": self.qrels_relevance_distribution or {},
                    "relevant_docs_per_query": self.qrels_relevant_docs_per_query or {},
                },
                "difficulty": self.difficulty_details,
            }
            export_path = os.path.join(self.output_dir, "dificultad_y_qrels.json")
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)


