import logging
from functools import lru_cache
from typing import Dict, Union, List, Any
from .pre_retrieval.idf import IDF
from .pre_retrieval.scq import SCQ
from .post_retrieval.wig import WIG
from .post_retrieval.nqc import NQC
from .post_retrieval.clarity import Clarity
from .post_retrieval.uef import UEF
from utils.text_processing import preprocess_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QPPMethodFactory:
    """
    Factory class for creating and managing QPP methods.
    Handles preprocessing and initialization of all QPP methods.
    """
    
    def __init__(self, index_builder, retrieval_results=None, rm_results=None, dataset_name=None):
        """
        Initialize the QPP method factory.
        
        Args:
            index_builder: IndexBuilder instance
            retrieval_results: DataFrame with retrieval results (for post-retrieval methods)
            rm_results: DataFrame with RM3 results (for UEF method)
            dataset_name: Name of the dataset (for language detection in preprocessing)
        """
        self.index = index_builder.index
        self.index_builder = index_builder
        self.retrieval_results = retrieval_results
        self.rm_results = rm_results
        self.dataset_name = dataset_name
        
        # Initialize QPP methods
        self._init_methods()
        
    def _init_methods(self):
        """Initialize all QPP methods."""
        # Pre-retrieval methods
        self.idf = IDF(self.index_builder)
        self.scq = SCQ(self.index_builder)
        
        # Post-retrieval methods (only if results are available)
        if self.retrieval_results is not None:
            self.wig = WIG(self.index_builder, self.retrieval_results)
            self.nqc = NQC(self.index_builder, self.retrieval_results)
            self.clarity = Clarity(self.index_builder, self.retrieval_results, dataset_name=self.dataset_name)
            
            if self.rm_results is not None:
                self.uef = UEF(self.index_builder, self.retrieval_results, self.rm_results)
    
    def compute_all_scores(self, queries: Dict[str, List[str]], **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Compute scores for all available QPP methods.
        
        Args:
            queries: Dictionary mapping query IDs to preprocessed query tokens
            **kwargs: Additional arguments including:
                - list_size_param: Default list size for methods without specific settings
                - wig_list_size: List size for WIG method (default: 5)
                - nqc_list_size: List size for NQC method (default: 100)
                - num_results: Minimum number of results required for valid queries (default: 1000)
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary mapping query IDs to their QPP scores
        """
        logger = logging.getLogger(__name__)
        
        # Get method-specific list sizes
        default_list_size = kwargs.get('list_size_param', 10)
        wig_list_size = kwargs.get('wig_list_size', 5)  # Default to 5 as per paper
        nqc_list_size = kwargs.get('nqc_list_size', 100)  # Default to 100 as per paper
        
        # Log list size settings
        logger.info(
            f"Using list sizes - WIG: {wig_list_size}, NQC: {nqc_list_size}, "
            f"Default: {default_list_size}"
        )
        
        # Get available query IDs from retrieval results
        available_qids = set(self.retrieval_results['qid'].astype(str).unique())
        input_qids = set(queries.keys())
        
        # Filter queries with too few results
        min_results = kwargs.get('num_results', 1000)  # Get minimum required results
        query_result_counts = self.retrieval_results.groupby('qid').size()
        valid_qids = set(query_result_counts[query_result_counts >= min_results * 0.1].index.astype(str))
        
        # Log query filtering
        logger.info(
            f"Processing queries: {len(input_qids)} input queries, "
            f"{len(available_qids)} have retrieval results, "
            f"{len(valid_qids)} have sufficient results (≥{min_results * 0.1:.0f})"
        )
        
        filtered_qids = available_qids - valid_qids
        if filtered_qids:
            logger.warning(
                f"Filtered out {len(filtered_qids)} queries with insufficient results: "
                f"{sorted(filtered_qids)[:5]}..."
            )
        
        # Use only valid queries that were in the input
        valid_query_ids = valid_qids & input_qids
        valid_queries = {qid: queries[qid] for qid in valid_query_ids}  # Create dictionary instead of tuple
        
        scores = {}
        
        # Initialize scores dict for all input queries
        for qid in input_qids:
            scores[qid] = {
                'idf_avg': 0.0,
                'idf_max': 0.0,
                'scq_avg': 0.0,
                'scq_max': 0.0
            }
        
        # Compute pre-retrieval scores for valid queries
        valid_scores = {}
        for qid in valid_queries:
            valid_scores[qid] = {
                'idf_avg': self.idf.compute_scores_batch(valid_queries, method='avg').get(qid, 0.0),
                'idf_max': self.idf.compute_scores_batch(valid_queries, method='max').get(qid, 0.0),
                'scq_avg': self.scq.compute_scores_batch(valid_queries, method='avg').get(qid, 0.0),
                'scq_max': self.scq.compute_scores_batch(valid_queries, method='max').get(qid, 0.0)
            }
        
        # Update scores with valid results
        scores.update(valid_scores)
        
        # Add post-retrieval scores if available
        if hasattr(self, 'wig'):
            logger.info("Computing post-retrieval scores with method-specific list sizes")
            
            # Compute post-retrieval scores with different list sizes
            post_retrieval_scores = {
                'wig': self.wig.compute_scores_batch(valid_queries, list_size_param=wig_list_size),
                'nqc': self.nqc.compute_scores_batch(valid_queries, list_size_param=nqc_list_size),
                'clarity': self.clarity.compute_scores_batch(valid_queries)
            }
            
            # Update scores dictionary for queries with retrieval results
            for qid in input_qids:
                for method, score_dict in post_retrieval_scores.items():
                    scores[qid][method] = score_dict.get(qid, 0.0)
            
            # Add UEF scores if available
            if hasattr(self, 'uef'):
                logger.debug("Computing UEF scores")
                uef_scores = {
                    'uef_wig': self.uef.compute_scores_batch(valid_queries, post_retrieval_scores['wig'], wig_list_size),
                    'uef_nqc': self.uef.compute_scores_batch(valid_queries, post_retrieval_scores['nqc'], nqc_list_size),
                    'uef_clarity': self.uef.compute_scores_batch(valid_queries, post_retrieval_scores['clarity'])
                }
                
                for qid in input_qids:
                    for method, score_dict in uef_scores.items():
                        scores[qid][method] = score_dict.get(qid, 0.0)
        
        # Log summary statistics
        non_zero_queries = sum(1 for qid in scores if any(v > 0 for v in scores[qid].values()))
        logger.info(
            f"Computed scores for {len(scores)} queries, "
            f"{non_zero_queries} have non-zero scores "
            f"({non_zero_queries/len(scores)*100:.1f}%)"
        )
        
        return scores 