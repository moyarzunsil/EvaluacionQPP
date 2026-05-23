import numpy as np
import pandas as pd
from typing import Dict
from metodos.base import PostRetrievalMethod
from utils.config import DATASET_FORMATS
import logging


class UEF(PostRetrievalMethod):
    def __init__(self, index_builder, retrieval_results, rm_results_df=None, dataset_name="antique_test"):
        super().__init__(index_builder, retrieval_results)
        # Clean the dataframes to standardize column names
        self.retrieval_results = self._clean_dataframe(retrieval_results.copy())
        self.rm_results_df = self._clean_dataframe(rm_results_df.copy()) if rm_results_df is not None else None
        # Define expected column names after cleaning
        self.query_id_col = 'query_id'
        self.doc_id_col = 'doc_id'
        self.score_col = 'score'
        # Validate columns after cleaning
        self._validate_columns(self.retrieval_results, 'retrieval_results')
        if self.rm_results_df is not None:
            self._validate_columns(self.rm_results_df, 'rm_results_df')

    def _validate_columns(self, df, df_name):
        """Ensure required columns exist"""
        required = [self.query_id_col, self.doc_id_col, self.score_col]
        if not all(col in df.columns for col in required):
            raise ValueError(f"Missing columns in {df_name}: {required}")        

    def _clean_dataframe(self, df):
        """Standardize dataframe columns using dataset config"""
        column_mapping = {
            'qid': 'query_id',     # Maps 'qid' to 'query_id'
            'docno': 'doc_id',     # Maps 'docno' to 'doc_id'
            'docScore': 'score'    # Maps 'docScore' to 'score'
        }
        return df.rename(columns=column_mapping)

    def compute_score(self, qid: str, predictor_score: float) -> float:
        """
        Compute UEF score for a single query.
        
        Args:
            qid: Query ID
            predictor_score: Score from the base predictor (WIG/NQC)
        
        Returns:
            float: UEF score, or 0.0 if correlation cannot be computed
        """
        logger = logging.getLogger(__name__)
        
        # Get results for this query
        original_results = self.retrieval_results[
            self.retrieval_results[self.query_id_col] == qid
        ].copy()
        
        rm_results = self.rm_results_df[
            self.rm_results_df[self.query_id_col] == qid
        ].copy()
        
        if original_results.empty or rm_results.empty:
            logger.debug(f"No results found for query {qid}")
            return 0.0

        try:
            # Convert to score series
            orig_scores = original_results.set_index(self.doc_id_col)[self.score_col]
            rm_scores = rm_results.set_index(self.doc_id_col)[self.score_col]
            
            # Calculate correlation
            common_docs = orig_scores.index.intersection(rm_scores.index)
            if len(common_docs) < 2:
                logger.debug(f"Insufficient common documents ({len(common_docs)}) for query {qid}")
                return 0.0
            
            # Normalize scores to prevent numerical issues
            orig_norm = (orig_scores[common_docs] - orig_scores[common_docs].mean()) / orig_scores[common_docs].std()
            rm_norm = (rm_scores[common_docs] - rm_scores[common_docs].mean()) / rm_scores[common_docs].std()
            
            # Handle zero standard deviation
            if orig_norm.isna().any() or rm_norm.isna().any():
                logger.debug(f"Zero variance in scores for query {qid}")
                return 0.0
            
            correlation = orig_norm.corr(rm_norm)
            
            # Handle NaN correlation
            if pd.isna(correlation):
                logger.debug(f"NaN correlation for query {qid}")
                return 0.0
            
            return correlation * predictor_score
            
        except Exception as e:
            logger.error(f"Error computing UEF score for query {qid}: {e}")
            return 0.0

    def compute_scores_batch(self, processed_queries: Dict[str, list], 
                            predictor_scores: Dict[str, float], 
                            list_size: int = None) -> Dict[str, float]:
        """
        Batch implementation using compute_score.
        
        Args:
            processed_queries: Dictionary mapping query IDs to preprocessed query tokens
            predictor_scores: Dictionary mapping query IDs to their predictor scores (WIG/NQC)
            list_size: Size of result list to consider (should match the predictor's list size)
                      If None, uses all available results
        
        Returns:
            Dict[str, float]: Dictionary mapping query IDs to their UEF scores
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Computing UEF scores with list size: {list_size}")
        
        if list_size is not None and list_size < 5:
            logger.warning(f"List size {list_size} may be too small for reliable UEF scores")
        
        uef_scores = {}
        for qid in processed_queries.keys():
            try:
                # Filter results to top-k if list_size is specified
                if list_size is not None:
                    original_results = self.retrieval_results[
                        self.retrieval_results[self.query_id_col] == qid
                    ].head(list_size)
                    
                    rm_results = self.rm_results_df[
                        self.rm_results_df[self.query_id_col] == qid
                    ].head(list_size)
                    
                    # Check if we have enough results after filtering
                    if len(original_results) < 2 or len(rm_results) < 2:
                        logger.debug(f"Insufficient results after filtering for query {qid}")
                        uef_scores[qid] = 0.0
                        continue
                else:
                    original_results = self.retrieval_results[
                        self.retrieval_results[self.query_id_col] == qid
                    ]
                    rm_results = self.rm_results_df[
                        self.rm_results_df[self.query_id_col] == qid
                    ]
                
                # Set results in temporary dataframes
                temp_retrieval_results = self.retrieval_results.copy()
                temp_rm_results = self.rm_results_df.copy()
                
                # Update instance temporarily for compute_score
                self.retrieval_results = original_results
                self.rm_results_df = rm_results
                
                # Compute score
                score = self.compute_score(qid, predictor_scores.get(qid, 0.0))
                uef_scores[qid] = score
                
                # Restore original dataframes
                self.retrieval_results = temp_retrieval_results
                self.rm_results_df = temp_rm_results
                
            except Exception as e:
                logger.error(f"Error computing UEF score for query {qid}: {e}")
                uef_scores[qid] = 0.0
        
        # Log statistics about computed scores
        non_zero_scores = sum(1 for score in uef_scores.values() if score != 0)
        logger.info(f"Computed {len(uef_scores)} UEF scores, {non_zero_scores} non-zero")
        
        return uef_scores