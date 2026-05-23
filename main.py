import nltk
import argparse
import sys
import os
import pyterrier as pt
import pandas as pd 
import logging

from data.dataset_processor import DatasetProcessor
from indexing.index_builder import IndexBuilder
from evaluation.qrels_difficulty_analyzer import QrelsDifficultyAnalyzer
from metodos.qpp_factory import QPPMethodFactory
from retrieval.retrieval import get_batch_scores
from evaluation.evaluator import evaluate_results
from evaluation.correlation_analyzer import QPPCorrelationAnalyzer
from utils.config import AVAILABLE_DATASETS

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt')

AVAILABLE_METRICS = ['ndcg@5', 'ndcg@10', 'ndcg@20', 'ap']
AVAILABLE_CORRELATIONS = ['kendall', 'spearman', 'pearson']

def setup_logging(log_file='loaded_index.log'):
    """Configure logging to write to both file and stderr."""
    # Remove any existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        
    # Create formatters and handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        # Try to create log file in script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(script_dir, 'logs')
        os.makedirs(log_path, exist_ok=True)
        log_file_path = os.path.join(log_path, log_file)

        # File handler
        file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        print(f"Logging to file: {log_file_path}")

    except Exception as e:
        print(f"Warning: Could not create log file ({e}). Logging to console only.")
    
    # Stream handler that writes to sys.stdout (instead of stderr)
    class StdoutStreamHandler(logging.StreamHandler):
        def __init__(self):
            super().__init__(sys.stdout)
    
    stream_handler = StdoutStreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    
    # Configure root logger
    root.setLevel(logging.INFO)
    
    # Disable other loggers that might be too verbose
    logging.getLogger('pyterrier').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

def process_dataset(dataset_name: str, dataset_path: str, args) -> QPPCorrelationAnalyzer:
    """Process a single dataset and return its correlation analyzer."""
    print(f"\nProcessing dataset: {dataset_name}")
    
    if not pt.started():
        pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
    
    try:
        dataset_processor = DatasetProcessor(dataset_path)
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        return None
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(script_dir, "indices", dataset_name)
    
    # Build or load index
    index_builder = IndexBuilder(dataset_processor, dataset_name)
    index = index_builder.load_or_build_index(index_path)
    
    # Save term frequencies for analysis (as in old version)
    sample_file_path = os.path.join(script_dir, "sample_term_frequencies.json")
    index_builder.save_sample_frequencies_to_json(sample_file_path)
    
    # Get raw queries for retrieval
    try:
        raw_queries = dataset_processor.get_raw_queries()  # Get only raw queries for retrieval
        if args.max_queries:
            query_ids = list(raw_queries.keys())[:args.max_queries]
            raw_queries = {qid: raw_queries[qid] for qid in query_ids}
    except ValueError as e:
        print(f"Error getting queries: {e}")
        return None
    
    # Prepare queries DataFrame for retrieval
    queries_df = pd.DataFrame(list(raw_queries.items()), columns=['qid', 'query'])
    
    # Get preprocessed queries for QPP methods
    processed_queries = dataset_processor.get_processed_queries()
    if args.max_queries:
        processed_queries = {qid: processed_queries[qid] for qid in query_ids}
    
    # Get retrieval results with num_results parameter
    retrieval_results = get_batch_scores(
        queries_df=queries_df,
        index=index,
        dataset=dataset_processor.dataset,
        method='BM25',
        num_results=args.num_results,  # Pass num_results parameter
        controls={
            'BM25': {'k1': 1.5, 'b': 0.8}
        }
    )
    
    # After getting retrieval results
    print(f"\nRetrieval results for {dataset_name}:")
    print(f"Columns: {retrieval_results.columns.tolist()}")
    print(f"Sample result:\n{retrieval_results.head(1).to_dict('records')}")
    
    # Get RM3 results for UEF method (matching old version's parameters)
    rm_results_df = None
    if args.use_uef:
        rm_results_df = get_batch_scores(
            queries_df=queries_df,
            index=index,
            dataset=dataset_processor.dataset,
            method='RM3',
            num_results=1000,  # Fixed value as in old version
            controls={
                'RM3': {
                    'fb_terms': 10,
                    'fb_docs': 10,
                    'original_weight': 0.5
                }
            }
        )
        print(f"RM3 columns: {rm_results_df.columns.tolist()}")
    
    # Get qrels
    qrels = dataset_processor.get_qrels()
    
    # Use both nDCG and AP metrics by default
    evaluation_metrics = ['ndcg@10', 'ap']
    if args.metrics:  # Add any additional metrics requested, but only if they're nDCG or AP
        additional_metrics = [m for m in args.metrics 
                            if m not in evaluation_metrics and 
                            (m.startswith('ndcg@') or m == 'ap')]
        evaluation_metrics.extend(additional_metrics)
    
    # Asegurar que las métricas para dificultad estén incluidas si son compatibles
    difficulty_metrics = getattr(args, "difficulty_metric", None)
    valid_difficulty_metrics = []
    if difficulty_metrics:
        for m in difficulty_metrics:
            if m == 'ap' or m.startswith('ndcg@'):
                valid_difficulty_metrics.append(m)
                if m not in evaluation_metrics:
                    evaluation_metrics.append(m)
            else:
                print(f"WARNING: difficulty metric '{m}' is not supported. "
                      "Use 'ap' o 'ndcg@k'. Ignorando esta métrica.")
        if not valid_difficulty_metrics:
            difficulty_metrics = None
        else:
            difficulty_metrics = valid_difficulty_metrics

    # Evaluate results with num_results parameter
    evaluation_results = evaluate_results(
        qrels_df=qrels,
        results_df=retrieval_results,
        metrics=evaluation_metrics,
        output_dir=os.path.join(script_dir, "evaluation_results", dataset_name),
        dataset_name=dataset_name,
        min_results=args.num_results  # Pass num_results parameter
    )
    
    # Print evaluation results (as in old version)
    print("\nRetrieval Evaluation Results:")
    for metric, scores in evaluation_results.items():
        print(f"\n{metric.upper()}:")
        print(f"Mean: {scores['mean']:.4f}")
        print("Sample of per-query scores:")
        for qid, score in list(scores['per_query'].items())[:5]:
            print(f"  Query {qid}: {score:.4f}")
    
    # Before creating QPP factory
    if 'text' not in retrieval_results.columns:
        print("WARNING: 'text' field missing from retrieval results!")
    
    # Análisis adicional de qrels y dificultad de consulta
    if not args.skip_plots:
        qrels_output_dir = os.path.join(
            script_dir, "evaluation_results", dataset_name
        )
        qrels_analyzer = QrelsDifficultyAnalyzer(
            qrels_df=qrels,
            evaluation_results=evaluation_results,
            dataset_name=dataset_name,
            output_dir=qrels_output_dir,
            difficulty_metrics=difficulty_metrics
        )
        qrels_analyzer.generate_all_plots()
    
    # Create QPP factory with preprocessed queries
    qpp_factory = QPPMethodFactory(
        index_builder=index_builder,
        retrieval_results=retrieval_results,
        rm_results=rm_results_df,
        dataset_name=dataset_path
    )
    
    # Compute QPP scores with preprocessed queries
    qpp_scores = qpp_factory.compute_all_scores(
        queries=processed_queries,
        list_size_param=args.list_size,
        wig_list_size=args.wig_list_size,
        nqc_list_size=args.nqc_list_size,
        num_results=args.num_results
    )
    
    # Create and return correlation analyzer with DPI settings
    return QPPCorrelationAnalyzer(
        qpp_scores=qpp_scores,
        retrieval_metrics=evaluation_results,
        output_dir=os.path.join(script_dir, "correlation_analysis", dataset_name),
        dpi=300  # Add high DPI setting for better quality images
    )

def main():
    parser = argparse.ArgumentParser(description='Run QPP evaluation on specified datasets')
    
    # Add logging configuration
    parser.add_argument('--log-file', type=str, default='loaded_index.log',
                       help='Log file name (will be created in logs/ directory)')
    
    # Dataset selection
    parser.add_argument('--datasets', nargs='+', choices=AVAILABLE_DATASETS.keys(),
                       default=list(AVAILABLE_DATASETS.keys()),
                       help='Datasets to process (default: all)')
    
    # Query processing
    parser.add_argument('--max-queries', type=int, default=None,
                       help='Maximum number of queries to process per dataset')
    parser.add_argument('--list-size', type=int, default=10,
                       help='List size parameter for QPP methods (default: 10)')
    parser.add_argument('--num-results', type=int, default=1000,
                       help='Number of results to retrieve (default: 1000)')
    
    # Evaluation options
    parser.add_argument('--metrics', nargs='+', choices=AVAILABLE_METRICS,
                       default=['ndcg@10', 'ap'],
                       help='Evaluation metrics to use (default: ndcg@10 and ap)')
    parser.add_argument('--correlations', nargs='+', choices=AVAILABLE_CORRELATIONS,
                       default=['kendall'],
                       help='Correlation coefficients to compute (default: kendall)')
    
    # Analysis options
    # Español: UEF se habilita por defecto; el flag se mantiene por compatibilidad.
    parser.add_argument('--use-uef', action='store_true', default=True,
                       help='Enable UEF-based QPP methods (enabled by default)')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory for results')
    parser.add_argument('--difficulty-metric', nargs='+', default=None,
                       help="Metrics to define query difficulty (e.g. 'ap' or 'ndcg@10'). "
                            "Default: all available metrics.")
    
    # Add new arguments for method-specific list sizes
    parser.add_argument('--wig-list-size', type=int, default=5,
                       help='List size parameter for WIG method (default: 5)')
    parser.add_argument('--nqc-list-size', type=int, default=200,
                       help='List size parameter for NQC method (default: 200)')
    
    args = parser.parse_args()
    
    # Setup logging with specified file
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    logger.info("Starting QPP evaluation...")
    
    # Create output directories if they don't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Process datasets with both PDF and PNG output
    dataset_analyzers = {}
    for dataset_name in args.datasets:
        dataset_path = AVAILABLE_DATASETS[dataset_name]
        analyzer = process_dataset(dataset_name, dataset_path, args)
        if analyzer:
            dataset_analyzers[dataset_name] = analyzer
            
            if not args.skip_plots:
                # Generate individual dataset report and plots (now in both PDF and PNG)
                analyzer.generate_report(args.correlations)
                for corr_type in args.correlations:
                    analyzer.plot_correlations_boxplot(corr_type)
    
    # Generate cross-dataset analysis if multiple datasets
    if len(dataset_analyzers) > 1 and not args.skip_plots:
        output_dir = args.output_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "correlation_analysis"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        for corr_type in args.correlations:
            QPPCorrelationAnalyzer.plot_correlations_across_datasets(
                datasets=dataset_analyzers,
                correlation_type=corr_type,
                output_dir=output_dir
            )

if __name__ == "__main__":
    main()