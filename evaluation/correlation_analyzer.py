import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional
import logging
import os
from utils.file_utils import ensure_dir

# Configuración estética global
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5)
plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'savefig.bbox': 'tight'
})

METHOD_NAME_MAP = {
    'idf_avg': 'IDF Average',
    'idf_max': 'IDF Maximum',
    'scq_avg': 'SCQ Average',
    'scq_max': 'SCQ Maximum',
    'wig': 'WIG',
    'nqc': 'NQC',
    'clarity': 'Clarity',
    'uef_wig': 'UEF-WIG',
    'uef_nqc': 'UEF-NQC',
    'uef_clarity': 'UEF-Clarity'
}

# Mapeo simple para nombres de métricas en gráficos/leyendas
METRIC_LABEL_MAP = {
    'ap': 'AP',   # Average Precision por consulta
    'map': 'MAP',
}

class QPPCorrelationAnalyzer:
    """
    Analyzes correlations between QPP predictions and retrieval effectiveness metrics (nDCG and AP).
    """
    
    def __init__(self, qpp_scores: Dict[str, Dict[str, float]], 
                 retrieval_metrics: Dict[str, Dict[str, Union[float, Dict[str, float]]]],
                 output_dir: Optional[str] = None,
                 dpi: int = 300):
        """
        Initialize the QPP correlation analyzer.
        
        Args:
            qpp_scores: Dictionary of QPP scores {qid: {method: score}}
            retrieval_metrics: Dictionary of retrieval metrics {metric: {per_query: {qid: score}, mean: float}}
                             Only nDCG and AP metrics are supported
            output_dir: Directory to save results and plots
            dpi: DPI (dots per inch) for saved plots
        
        Raises:
            ValueError: If qpp_scores is empty or retrieval_metrics contains no valid metrics
        """
        self.logger = logging.getLogger(__name__)
        self.dpi = dpi

        # Validate input data
        if not qpp_scores:
            raise ValueError("QPP scores dictionary cannot be empty")
        
        # Validate metrics - only allow nDCG and AP
        valid_metrics = {k: v for k, v in retrieval_metrics.items() 
                        if any(k.lower().startswith(prefix) 
                              for prefix in ['ndcg@', 'map', 'ap', 'p@', 'rr@'])}
        
        if not valid_metrics:
            raise ValueError("No valid metrics found. Supported metrics: nDCG@k, AP/MAP, P@k, RR@k")
            
        if len(valid_metrics) != len(retrieval_metrics):
            self.logger.warning("Some metrics were filtered out. Only nDCG@k and AP metrics are supported.")
        
        # Add validation for QPP scores
        for method, scores in qpp_scores.items():
            if not all(isinstance(score, (int, float)) for score in scores.values()):
                raise ValueError(f"Non-numeric QPP scores found in method: {method}")
                
        self.qpp_scores = qpp_scores
        self.retrieval_metrics = valid_metrics
        self.output_dir = ensure_dir(output_dir) if output_dir else None
        # Align query sets between QPP and metrics
        common_qids = set(qpp_scores.keys()) & set(retrieval_metrics['ndcg@10']['per_query'].keys())
        
        self.qpp_scores = {k:v for k,v in qpp_scores.items() if k in common_qids}
        self.retrieval_metrics = {
            metric: {
                'per_query': {k:v for k,v in scores['per_query'].items() if k in common_qids},
                'mean': np.mean(list(scores['per_query'].values()))
            } 
            for metric, scores in retrieval_metrics.items()
        }
        
        # Convert data to DataFrames for easier analysis
        self.qpp_df = pd.DataFrame.from_dict(qpp_scores, orient='index')
        self.metrics_df = pd.DataFrame({
            metric: scores['per_query'] 
            for metric, scores in valid_metrics.items()
        })
        
        # Align QIDs between QPP scores and metrics
        self.align_qids()

    def calculate_correlations(self, 
                         correlation_types: List[str] = ['pearson', 'spearman', 'kendall'],
                         min_queries: int = 5,
                         min_results_per_query: Optional[int] = None,
                         return_pvalues: bool = False) -> Union[Dict[str, pd.DataFrame], tuple]:
        """
        Calculate correlations between QPP scores and retrieval metrics.

        Args:
            return_pvalues: If True, returns a tuple (correlations, p_values)
        """
        correlations = {}
        p_values = {} if return_pvalues else None

        for corr_type in correlation_types:
            corr_df = pd.DataFrame(index=self.qpp_df.columns, columns=self.metrics_df.columns)
            pval_df = pd.DataFrame(index=self.qpp_df.columns, columns=self.metrics_df.columns) if return_pvalues else None

            for qpp_method in self.qpp_df.columns:
                for metric in self.metrics_df.columns:
                    valid_mask = ~(self.qpp_df[qpp_method].isna() | self.metrics_df[metric].isna())
                    x = self.qpp_df[qpp_method][valid_mask]
                    y = self.metrics_df[metric][valid_mask]

                    try:
                        if len(x) >= min_queries:
                            if corr_type == 'pearson':
                                corr, pval = stats.pearsonr(x, y)
                            elif corr_type == 'spearman':
                                corr, pval = stats.spearmanr(x, y)
                            else:  # kendall
                                corr, pval = stats.kendalltau(x, y)
                            
                            corr_df.loc[qpp_method, metric] = corr
                            if return_pvalues:
                                pval_df.loc[qpp_method, metric] = pval

                            # Logging (mantener existente)
                        else:
                            corr_df.loc[qpp_method, metric] = float('nan')
                            if return_pvalues:
                                pval_df.loc[qpp_method, metric] = float('nan')
                    except Exception as e:
                        corr_df.loc[qpp_method, metric] = float('nan')
                        if return_pvalues:
                            pval_df.loc[qpp_method, metric] = float('nan')

            correlations[corr_type] = corr_df
            if return_pvalues:
                p_values[corr_type] = pval_df

        return (correlations, p_values) if return_pvalues else correlations
    
    def plot_pvalue_heatmap(self, correlation_type: str = 'kendall', save_plot: bool = True) -> None:
        """
        Enhanced heatmap visualization of statistical significance with better categorical 
        representation and improved visual hierarchy.
        """
        # Get p-values matrix
        result = self.calculate_correlations([correlation_type], return_pvalues=True)
        _, p_values = result
        pval_df = p_values[correlation_type]
        
        # Preprocess p-values
        pval_df = pval_df.apply(pd.to_numeric, errors='coerce').fillna(1)
        pval_df = pval_df.replace(0.0, np.finfo(float).tiny)  # Avoid log(0)

        # Usar etiquetas amigables para las métricas (AP/MAP)
        pval_df.columns = [METRIC_LABEL_MAP.get(c, c) for c in pval_df.columns]
        
        # Convert to Spanish method names
        pval_df.index = pval_df.index.map(lambda x: METHOD_NAME_MAP.get(x, x))
        
        # Create significance categories
        significance_bins = [1, 0.05, 0.01, 0.001, 0]
        significance_labels = [
            'Not significant (≥0.05)',
            'Significant (<0.05)',
            'Very significant (<0.01)',
            'Highly significant (<0.001)'
        ]
        
        # Create annotated matrix with stars and categories
        annot = pd.DataFrame(index=pval_df.index, columns=pval_df.columns)
        category_matrix = pd.DataFrame(index=pval_df.index, columns=pval_df.columns)
        
        for col in pval_df.columns:
            for idx in pval_df.index:
                p = pval_df.loc[idx, col]
                if p >= 0.05:
                    annot.loc[idx, col] = '≥0.05'
                    category = 0
                elif 0.01 <= p < 0.05:
                    annot.loc[idx, col] = '<0.05'
                    category = 1
                elif 0.001 <= p < 0.01:
                    annot.loc[idx, col] = '<0.01'
                    category = 2
                else:
                    annot.loc[idx, col] = '<0.001'  # For p < 0.001
                    category = 3
                category_matrix.loc[idx, col] = category

        # Create visualization
        plt.figure(figsize=(14, 10))
        ax = plt.gca()
        
        # Custom colormap: white -> yellow -> orange -> red
        cmap = sns.color_palette("rocket_r", n_colors=4)
        
        sns.heatmap(
            category_matrix.astype(int),
            annot=annot,
            fmt='',
            cmap=cmap,
            cbar=False,
            linewidths=0.5,
            linecolor='lightgray',
            annot_kws={'fontsize': 16, 'color': 'white', 'weight': 'bold'}
        )
        
        # Add color bar with labels
        cbar = ax.figure.colorbar(
            ax.collections[0],
            ticks=[0.375, 1.125, 1.875, 2.625],
            shrink=0.8
        )
        cbar.ax.set_yticklabels(significance_labels)
        cbar.ax.tick_params(labelsize=11)
        cbar.ax.set_title('Significance level', fontsize=12)
        
        # Add significance threshold line
        plt.axhline(y=0, color='white', linewidth=3, xmin=0, xmax=1)
        
        # Formatting
        plt.title(f'Statistical Significance - {correlation_type.capitalize()}\n', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Retrieval Metrics', fontsize=14)
        plt.ylabel('QPP Methods', fontsize=14)
        plt.tight_layout()
        
        # Save/show plot
        if save_plot and self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'pvalues_qpp_{correlation_type}.pdf'), dpi=self.dpi)
            plt.savefig(os.path.join(self.output_dir, f'pvalues_qpp_{correlation_type}.png'), dpi=self.dpi)
            plt.close()
        else:
            plt.show()

    def plot_correlation_heatmap(self, correlation_type: str = 'kendall', 
                               save_plot: bool = True) -> None:
        """
        Plot heatmap of correlations between QPP methods and retrieval metrics.
        
        Args:
            correlation_type: Type of correlation to plot
            save_plot: Whether to save the plot to file
        """
        correlations = self.calculate_correlations([correlation_type])[correlation_type]
        
        # Convert to numeric, replacing any non-numeric values with NaN
        correlations = correlations.apply(pd.to_numeric, errors='coerce')
        
        # Map method names to Spanish
        correlations.index = correlations.index.map(lambda x: METHOD_NAME_MAP.get(x, x))
        # Map metric names to etiquetas legibles (AP/MAP)
        correlations.columns = [METRIC_LABEL_MAP.get(c, c) for c in correlations.columns]
        
        plt.figure(figsize=(12, 10))
        # Create mask for NaN values
        mask = np.isnan(correlations)
        
        sns.heatmap(
            correlations,
            annot=True,
            annot_kws={"size": 12},
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            fmt='.4f',
            linewidths=0.5,
            square=True,
            mask=mask
        )
        plt.title(f'{correlation_type.capitalize()} Correlation between QPP Methods and Metrics', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_plot and self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'correlacion_qpp_{correlation_type}.pdf'), dpi=self.dpi)
            plt.savefig(os.path.join(self.output_dir, f'correlacion_qpp_{correlation_type}.png'), dpi=self.dpi)
            plt.close()
        else:
            plt.show()

    def plot_correlation_heatmap_horizontal(self, correlation_type: str = 'kendall', 
                                          save_plot: bool = True) -> None:
        """
        Horizontal version of correlation heatmap optimized for thesis formatting.
        """
        correlations = self.calculate_correlations([correlation_type])[correlation_type]
        correlations = correlations.apply(pd.to_numeric, errors='coerce')
        
        # Transpose matrix for horizontal layout
        correlations = correlations.T  
        
        # Map method names to Spanish
        correlations.columns = correlations.columns.map(lambda x: METHOD_NAME_MAP.get(x, x))
        # Map metric names (ahora en índice) a etiquetas legibles
        correlations.index = [METRIC_LABEL_MAP.get(i, i) for i in correlations.index]
        
        plt.figure(figsize=(15, 8))
        ax = sns.heatmap(
            correlations,
            annot=True,
            annot_kws={"size": 11},
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            fmt='.4f',
            linewidths=0.5,
            cbar_kws={'label': f'Correlación {correlation_type}', 'shrink': 0.8},
            square=False
        )
        
        # Rotate metric labels on x-axis
        plt.xticks(rotation=45, ha='right', fontsize=12)
        
        # Adjust method labels on y-axis
        plt.yticks(rotation=0, fontsize=12, va='center')
        
        # Adjust title and labels for horizontal layout
        ax.set_title(f'{correlation_type.capitalize()} Correlations - QPP Methods vs Metrics\n', 
                   fontsize=14, pad=20)
        ax.set_xlabel('QPP Methods', fontsize=12)
        ax.set_ylabel('Retrieval Metrics', fontsize=12)
        
        # Move colorbar to bottom
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel(f'{correlation_type.capitalize()} Correlation', rotation=-90, va="bottom", labelpad=15)
        cbar.ax.tick_params(labelsize=10)
        
        plt.tight_layout()
        
        if save_plot and self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'correlacion_horizontal_{correlation_type}.pdf'), 
                      dpi=self.dpi, bbox_inches='tight')
            plt.savefig(os.path.join(self.output_dir, f'correlacion_horizontal_{correlation_type}.png'), 
                      dpi=self.dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_scatter_plots(self, metric: str, save_plots: bool = True) -> None:
        """
        Create scatter plots between each QPP method and a specific retrieval metric.
        
        Args:
            metric: The retrieval metric to plot against
            save_plots: Whether to save the plots to files
        """
        n_methods = len(self.qpp_df.columns)
        metric_label = METRIC_LABEL_MAP.get(metric, metric)
        fig, axes = plt.subplots(
            (n_methods + 2) // 3, 3,
            figsize=(15, 5 * ((n_methods + 2) // 3)),
            squeeze=False
        )
        
        for idx, qpp_method in enumerate(self.qpp_df.columns):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            sns.regplot(
                x=self.qpp_df[qpp_method],
                y=self.metrics_df[metric],
                ax=ax,
                scatter_kws={'alpha': 0.6, 'color': '#3498db'},
                line_kws={'color': '#e67e22', 'linewidth': 2}
            )
            
            corr, _ = stats.kendalltau(self.qpp_df[qpp_method], self.metrics_df[metric])
            method_name = METHOD_NAME_MAP.get(qpp_method, qpp_method)
            ax.set_title(f'{method_name}\nτ = {corr:.4f}', fontsize=14, pad=10)
            ax.set_xlabel('QPP Score')
            ax.set_ylabel(f'{metric_label} Score')
            
        # Remove empty subplots
        for idx in range(n_methods, len(axes.flat)):
            fig.delaxes(axes.flat[idx])
            
        plt.tight_layout()
        
        if save_plots and self.output_dir:
            # Guardar gráfico combinado
            combined_path = os.path.join(self.output_dir, f'dispersion_qpp_{metric}')
            plt.savefig(f'{combined_path}.pdf', dpi=self.dpi)
            plt.savefig(f'{combined_path}.png', dpi=self.dpi)
            plt.close()
            
            # Crear subcarpeta para gráficos individuales
            scatter_dir = os.path.join(self.output_dir, 'scatter_individual')
            ensure_dir(scatter_dir)
            
            # Generar gráficos individuales
            for qpp_method in self.qpp_df.columns:
                plt.figure(figsize=(8, 6))
                ax = plt.gca()
                
                sns.regplot(
                    x=self.qpp_df[qpp_method],
                    y=self.metrics_df[metric],
                    ax=ax,
                    scatter_kws={'alpha': 0.6, 'color': '#3498db'},
                    line_kws={'color': '#e67e22', 'linewidth': 2}
                )
                
                corr, _ = stats.kendalltau(self.qpp_df[qpp_method], self.metrics_df[metric])
                method_name = METHOD_NAME_MAP.get(qpp_method, qpp_method)
                ax.set_title(f'{method_name}\nτ = {corr:.4f}', fontsize=14, pad=10)
                ax.set_xlabel('QPP Score')
                ax.set_ylabel(f'{metric_label} Score')
                plt.tight_layout()
                
                # Guardar gráfico individual
                filename = f'scatter_{metric}_{qpp_method}'
                plt.savefig(
                    os.path.join(scatter_dir, f'{filename}.png'), 
                    dpi=self.dpi
                )
                plt.close()
                
        else:
            plt.show()

    def plot_correlations_boxplot(self, correlation_type: str = 'kendall', save_plot: bool = True) -> None:
        # Get correlations
        correlations = self.calculate_correlations([correlation_type])[correlation_type]
        
        # Convert to numeric, replacing any remaining non-numeric values with NaN
        correlations = correlations.apply(pd.to_numeric, errors='coerce')
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Sort methods by median correlation value
        method_medians = correlations.median(axis=1)
        sorted_methods = method_medians.sort_values().index.tolist()
        
        # Prepare data for boxplot
        plot_data = []
        labels = []
        
        for method in sorted_methods:
            data = correlations.loc[method].dropna()
            if len(data) > 0:  # Only include methods with valid data
                plot_data.append(data)
                labels.append(method)
        
        # Map method names to Spanish
        labels = [METHOD_NAME_MAP.get(method, method) for method in labels]
        
        # Create boxplot
        bp = plt.boxplot(plot_data, labels=labels, patch_artist=True, widths=0.6)
        
        # Customize boxplot colors
        for box in bp['boxes']:
            box.set(facecolor='#2ecc71', linewidth=2, alpha=0.7)
        
        # Dibujar los puntos individuales
        for i, data in enumerate(plot_data):
            x = np.random.normal(i + 1, 0.04, size=len(data))  # Pequeño jitter en el eje X
            plt.plot(x, data, 'o', color='#3498db', alpha=0.6)  # Puntos azules con transparencia
        
        # Customize plot
        plt.ylabel(f'{correlation_type.capitalize()} Correlation', labelpad=15)
        plt.xlabel('QPP Method', labelpad=15)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.title("Distribution of Correlations by QPP Method", fontsize=18, pad=20)
        plt.grid(True, axis='y')
        
        # Add horizontal line at y=0 for reference
        plt.axhline(0, color='#e74c3c', linestyle='--', alpha=0.8, linewidth=1.5)
        
        y_min = min([min(data) for data in plot_data]) - 0.1  # Add a small buffer below
        y_max = max([max(data) for data in plot_data]) + 0.1  # Add a small buffer above
        plt.ylim(y_min, y_max)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_plot and self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'correlaciones_qpp_boxplot_{correlation_type}.pdf'), dpi=self.dpi)
            plt.savefig(os.path.join(self.output_dir, f'correlaciones_qpp_boxplot_{correlation_type}.png'), dpi=self.dpi)
            plt.close()
        else:
            plt.show()

    def align_qids(self):
        """
        Align QIDs between QPP scores and retrieval metrics.
        """
        # Align QIDs
        common_qids = self.qpp_df.index.intersection(self.metrics_df.index)
        self.qpp_df = self.qpp_df.loc[common_qids]
        self.metrics_df = self.metrics_df.loc[common_qids]
        
        self.logger.info(f"Number of QIDs after alignment: {len(common_qids)}")
    
    def generate_report(self, correlation_types: List[str] = ['kendall']) -> None:
        """
        Generate a comprehensive correlation analysis report.

        Args:
            correlation_types: List of correlation types to include in report
        """
        if not self.output_dir:
            self.logger.warning("No output directory specified. Cannot save the report.")
            return
            
        correlations = self.calculate_correlations(correlation_types)

        with open(os.path.join(self.output_dir, 'qpp_correlation_report.txt'), 'w') as f:
            f.write("QPP Correlation Analysis Report\n")
            f.write("=====================================\n\n")
            f.write("Analyzed Metrics: nDCG@k and AP\n\n")
            
            for corr_type, corr_df in correlations.items():
                f.write(f"\nCorrelations {corr_type.upper()}:\n")
                f.write("------------------------\n")
                
                f.write("\nNumber of queries used:\n")
                for method in corr_df.index:
                    n_queries = len(self.qpp_df[method].dropna())
                    method_name = METHOD_NAME_MAP.get(method, method)
                    f.write(f"{method_name}: {n_queries} queries\n")
                
                f.write("\nCorrelation values (only statistically significant):\n")
                # Map method names for display
                display_df = corr_df.copy()
                display_df.index = display_df.index.map(lambda x: METHOD_NAME_MAP.get(x, x))
                f.write(display_df.to_string())
                f.write("\n\n")
                
                f.write("Summary Statistics:\n")
                best_method = corr_df.mean(axis=1).idxmax()
                best_metric = corr_df.mean(axis=0).idxmax()
                f.write(f"Best QPP method: {METHOD_NAME_MAP.get(best_method, best_method)}\n")
                f.write(f"Most predictable metric: {METRIC_LABEL_MAP.get(best_metric, best_metric)}\n")
                f.write("\n")
                
            # Generate plots
            for metric in self.metrics_df.columns:
                self.plot_scatter_plots(metric)
            
            for corr_type in correlation_types:
                self.plot_correlation_heatmap(corr_type)
                self.plot_correlation_heatmap_horizontal(corr_type)
                self.plot_correlations_boxplot(corr_type)
                self.plot_pvalue_heatmap(corr_type)