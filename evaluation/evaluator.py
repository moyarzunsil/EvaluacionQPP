import pandas as pd
import numpy as np
from typing import Dict, Union, Iterable, Optional
import os
import ir_measures
from ir_measures import nDCG, P, AP, RR, Judged
from utils.file_utils import ensure_dir
import logging
from utils.config import DATASET_FORMATS
import json
from evaluation.evaluator_viz import RetrievalMetricsVisualizer



def evaluate_results(
    qrels_df: pd.DataFrame,
    results_df: pd.DataFrame,
    metrics: list = ['ndcg@10', 'ap'],
    output_dir: Optional[str] = None,
    dataset_name: str = "antique_test",
    min_results: int = 1000
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Evaluate retrieval results using ir_measures with flexible column handling
    """
    dataset_config = DATASET_FORMATS.get(dataset_name, DATASET_FORMATS["antique_test"])
    
    # Handle column mappings with fallbacks
    default_qrels_columns = {'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'}
    default_run_columns = {'qid': 'query_id', 'docno': 'doc_id', 'docScore': 'score'}
    
    # Safe column renaming
    def safe_rename(df, column_map):
        return df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    # Process qrels and run with flexible column names
    qrels = safe_rename(qrels_df, dataset_config.get("qrels_columns", default_qrels_columns))
    run = safe_rename(results_df, dataset_config.get("run_columns", default_run_columns))

    # Ensure required columns exist
    required_qrels = ['query_id', 'doc_id', 'relevance']
    required_run = ['query_id', 'doc_id', 'score']
    
    for col in required_qrels:
        if col not in qrels.columns:
            raise ValueError(f"Missing required qrels column: {col}. Available columns: {qrels.columns.tolist()}")
    
    for col in required_run:
        if col not in run.columns:
            raise ValueError(f"Missing required run column: {col}. Available columns: {run.columns.tolist()}")
    
    # Document ID transformations
    if 'doc_id_transform' in dataset_config:
        transform = dataset_config["doc_id_transform"]
        qrels['doc_id'] = qrels['doc_id'].apply(transform)
        run['doc_id'] = run['doc_id'].apply(transform)

    # Ensure correct data types
    qrels['relevance'] = qrels['relevance'].astype(int)
    run['score'] = run['score'].astype(float)
    for col in ['query_id', 'doc_id']:
        qrels[col] = qrels[col].astype(str)
        run[col] = run[col].astype(str)

    # Query filtering
    valid_qids = set(qrels['query_id']).intersection(run['query_id'])
    qrels = qrels[qrels['query_id'].isin(valid_qids)]
    run = run[run['query_id'].isin(valid_qids)]
    
    if run.empty:
        return {metric: {'per_query': {}, 'mean': 0.0} for metric in metrics}

    # Sort run by score descending per query
    run = run.sort_values(['query_id', 'score'], ascending=[True, False])

    results_per_query = run.groupby('query_id').size()
    valid_qids = results_per_query[results_per_query >= min_results].index.astype(str)

    # Apply dual filtering (qrels intersection + min results)
    final_qids = valid_qids.intersection(qrels['query_id'].unique())
    qrels = qrels[qrels['query_id'].isin(final_qids)]
    run = run[run['query_id'].isin(final_qids)]

    # Log filtering results
    filtered_count = len(valid_qids) - len(final_qids)
    print(f"Filtered {filtered_count} queries with insufficient results "
        f"(<{min_results} docs) or missing qrels")
    print(f"Evaluating {len(final_qids)} queries with ≥{min_results} results")

    if run.empty:
        return {metric: {'per_query': {}, 'mean': 0.0} for metric in metrics}

    # Metric configuration
    metric_configs = []
    binary_threshold = dataset_config["binary_threshold"]
    gain_values = dataset_config["gain_values"]
    # Modify metric configuration section
    max_cutoff = max(
        [int(m.split('@')[1]) for m in metrics if m.startswith('ndcg@') or m.startswith('p@')],
        default=10
    )

    if min_results < max_cutoff:
        print(f"WARNING: min_results ({min_results}) < max metric cutoff ({max_cutoff}). "
            "Consider increasing num_results parameter")
    
    #print column names
    print("Available columns qrels:", qrels.columns.tolist())
    print("Available columns run:", run.columns.tolist())

    print("Sample run docs:", run['doc_id'].sample(3).values.tolist())  # Add .values
    print("Qrels docs:", qrels['doc_id'].sample(3).values.tolist())
    print("Qrels relevance distribution:", qrels['relevance'].value_counts())

    for metric in metrics:
        metric_lower = metric.lower()
        if metric_lower.startswith('ndcg@'):
            k = int(metric_lower.split('@')[1])
            gains = {int(k): int(v) for k, v in gain_values.items()}
            metric_configs.append((metric, ir_measures.nDCG(cutoff=k, gains=gains)))
        elif metric_lower == 'ap':
            metric_configs.append((metric, ir_measures.AP(rel=binary_threshold)))
        elif metric_lower.startswith('p@'):
            k = int(metric_lower.split('@')[1])
            metric_configs.append((metric, ir_measures.P(cutoff=k, rel=binary_threshold)))

    # Convert qrels to ir_measures compatible format
    qrels_dict = qrels.groupby('query_id').apply(
        lambda x: dict(zip(x['doc_id'], x['relevance']))
    ).to_dict()

    # Calculate metrics
    evaluator = ir_measures.evaluator([m for _, m in metric_configs], qrels_dict)
    all_results = list(evaluator.iter_calc(run))  # Pass run DataFrame directly
    
    # Process results per metric
    results = {}
    for name, measure in metric_configs:
        metric_results = [r for r in all_results if r.measure == measure]
        query_scores = {str(r.query_id): r.value for r in metric_results}
        results[name] = {
            'per_query': query_scores,
            'mean': np.mean(list(query_scores.values())) if query_scores else 0.0
        }

    # Output handling
    if output_dir:
        output_path = os.path.join(ensure_dir(output_dir), 'results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate visualizations
        visualizer = RetrievalMetricsVisualizer(results, output_dir, dataset_name)
        visualizer.generate_all_plots(save=True)
    
    return results