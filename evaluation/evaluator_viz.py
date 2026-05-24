import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import os
import re
import time
import json
from typing import Dict, Optional
from utils.file_utils import ensure_dir

class RetrievalMetricsVisualizer:
    """
    Visualizes retrieval evaluation metrics with various plots.
    Generates boxplots, histograms, mean metric bars, and scatter plots.
    """
    
    def __init__(self, eval_results: Dict, output_dir: Optional[str] = None, dataset_name: str = "unknown"):
        self.eval_results = eval_results
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.metrics_df = self._prepare_metrics_df()
        
    def _prepare_metrics_df(self) -> pd.DataFrame:
        """Convert per-query metrics into a DataFrame."""
        data = {}
        for metric, metric_data in self.eval_results.items():
            data[metric] = metric_data['per_query']
        return pd.DataFrame(data)
    
    def _format_metric_name(self, metric: str) -> str:
        """Format metric names for display in plots."""
        if metric.startswith('ndcg@'):
            k = metric.split('@')[1]
            return f'nDCG@{k}'
        elif metric == 'ap':
            # AP = Average Precision per query (according to ir_measures / MAP alias).
            return 'AP'
        elif metric.startswith('p@'):
            k = metric.split('@')[1]
            return f'P@{k}'
        elif metric.startswith('rr@'):
            k = metric.split('@')[1]
            return f'RR@{k}'
        else:
            return metric
    
    def _save_plot(self, filename: str):
        """Helper to save plots consistently.

        Ensures the directory exists, sanitizes the filename, and retries
        once if a transient writing error occurs (e.g., Windows).
        """
        if self.output_dir:
            ensure_dir(self.output_dir)
            safe_base = self._sanitize_filename(filename)
            path = os.path.normpath(os.path.join(self.output_dir, safe_base))

            def _attempt_save():
                plt.savefig(f'{path}.png', dpi=300, bbox_inches='tight')
                plt.savefig(f'{path}.pdf', bbox_inches='tight')

            try:
                _attempt_save()
            except OSError as e:
                # Reintento ligero por posibles bloqueos/latencias del FS en Windows
                time.sleep(0.25)
                _attempt_save()
            finally:
                plt.close()
        else:
            plt.show()

    def _sanitize_filename(self, name: str) -> str:
        """Sanitizes filenames for Windows/Linux.

        Replaces invalid characters and trailing spaces, and avoids reserved
        names.
        """
        # Replace invalid characters in Windows: <>:"/\|?*
        name = re.sub(r'[<>:"/\\|?*]+', '_', name)
        # Strip trailing dots/spaces that Windows does not allow
        name = name.rstrip(' .')
        # Avoid reserved names (CON, PRN, AUX, NUL, COM1.., LPT1..)
        reserved = {
            'CON','PRN','AUX','NUL','COM1','COM2','COM3','COM4','COM5','COM6','COM7','COM8','COM9',
            'LPT1','LPT2','LPT3','LPT4','LPT5','LPT6','LPT7','LPT8','LPT9'
        }
        base = os.path.basename(name)
        if base.upper() in reserved:
            name = f"_{name}"
        return name
    
    def plot_metric_distributions(self, save: bool = True):
        """Generate boxplots and histograms showing metric distributions."""
        # Boxplot
        plt.figure(figsize=(12, 8))
        df_formatted = self.metrics_df.rename(columns=self._format_metric_name)
        sns.boxplot(data=df_formatted, palette='viridis')
        plt.title('Distribution of Metrics per Query')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        if save:
            self._save_plot('boxplot_metricas')
            # Export JSON metadata for boxplot
            self._export_boxplot_metadata()
        else:
            plt.show()
        
        # Histograms
        n_metrics = len(self.metrics_df.columns)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.metrics_df.columns):
            ax = axes[i]
            formatted_name = self._format_metric_name(metric)
            sns.histplot(self.metrics_df[metric], kde=True, ax=ax, color='skyblue')
            ax.set_title(formatted_name)
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
        
        # Hide empty subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        if save:
            self._save_plot('histogramas_metricas')
        else:
            plt.show()

    def plot_mean_metrics(self, save: bool = True):
        """Bar plot showing mean values for each metric."""
        # For the mean of AP, it is common to refer to MAP (Mean Average Precision),
        # so we explicitly label that bar as MAP.
        means = {}
        for metric, data in self.eval_results.items():
            if metric == 'ap':
                label = 'MAP'
            else:
                label = self._format_metric_name(metric)
            means[label] = data['mean']
        metrics = list(means.keys())
        values = list(means.values())
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=metrics, y=values, palette='rocket')
        plt.title('Mean of Evaluation Metrics')
        plt.ylabel('Mean')
        plt.xticks(rotation=45)
        if save:
            self._save_plot('media_metricas')
            # Export JSON metadata for mean metrics
            self._export_mean_metrics_metadata(means)
        else:
            plt.show()

    def plot_metric_relationships(self, save: bool = True):
        """Scatter plots showing relationships between metrics."""
        metrics = self.metrics_df.columns.tolist()
        for i in range(len(metrics)):
            for j in range(i+1, len(metrics)):
                metric1 = metrics[i]
                metric2 = metrics[j]
                x = self.metrics_df[metric1]
                y = self.metrics_df[metric2]
                
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=x, y=y, alpha=0.6, color='#3498db')
                plt.title(f'Relationship between {self._format_metric_name(metric1)} and {self._format_metric_name(metric2)}')
                plt.xlabel(self._format_metric_name(metric1))
                plt.ylabel(self._format_metric_name(metric2))
                
                # Add correlation coefficient
                corr = stats.pearsonr(x, y)[0]
                plt.annotate(f'r = {corr:.2f}', 
                            xy=(0.05, 0.95), xycoords='axes fraction',
                            fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
                
                if save:
                    filename = f'scatter_{metric1}_vs_{metric2}'
                    self._save_plot(filename)
                else:
                    plt.show()

    def generate_all_plots(self, save: bool = True):
        """Generate all available visualizations."""
        self.plot_metric_distributions(save)
        self.plot_mean_metrics(save)
        self.plot_metric_relationships(save)

    def _export_boxplot_metadata(self):
        """Export JSON metadata for boxplot visualization."""
        if not self.output_dir:
            return
        
        metadata = {
            "dataset": self.dataset_name,
            "plot_type": "boxplot_metricas",
            "metrics": {}
        }
        
        for metric in self.metrics_df.columns:
            values = self.metrics_df[metric].dropna()
            formatted_name = self._format_metric_name(metric)
            metadata["metrics"][formatted_name] = {
                "count": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "q1": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q3": float(values.quantile(0.75)),
                "max": float(values.max()),
            }
        
        output_path = os.path.join(self.output_dir, "boxplot_metricas.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _export_mean_metrics_metadata(self, means: Dict[str, float]):
        """Export JSON metadata for mean metrics visualization."""
        if not self.output_dir:
            return
        
        metadata = {
            "dataset": self.dataset_name,
            "plot_type": "media_metricas",
            "mean_values": {k: float(v) for k, v in means.items()}
        }
        
        output_path = os.path.join(self.output_dir, "media_metricas.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)