import numpy as np
import pandas as pd
from typing import Dict, Iterable
from collections import defaultdict
from indexing.index_builder import IndexBuilder
from utils.text_processing import preprocess_text
from metodos.base import PostRetrievalMethod
import logging
import json

logger = logging.getLogger(__name__)

class Clarity(PostRetrievalMethod):
    def __init__(
        self,
        index_builder: IndexBuilder,
        retrieval_results: pd.DataFrame,
        dataset_name: str = None,
        top_k: int = 100,
        term_cutoff: int = 100,
        score_column: str = "docScore",
        mu_bg: float = 1000.0
    ):
        super().__init__(index_builder, retrieval_results, dataset_name)
        
        # Validate critical inputs
        if score_column not in retrieval_results.columns:
            raise ValueError(f"Retrieval results must contain '{score_column}' column")

        # Handle negative scores by shifting to non-negative range
        min_score = self.retrieval_results[score_column].min()
        if min_score < 0:
            print(f"Warning: Found negative retrieval scores (min: {min_score:.4f}). Shifting to non-negative range.")
            self.retrieval_results = self.retrieval_results.copy()
            self.retrieval_results[score_column] = self.retrieval_results[score_column] - min_score

        self.top_k = top_k
        self.term_cutoff = term_cutoff
        self.score_column = score_column
        self.index_stats = {
            'total_terms': index_builder.total_terms,
            'term_cf': index_builder.term_cf
        }
        self.dataset_name = dataset_name

        # Dirichlet smoothing configuration for background model P(w|C)
        # En colecciones pequeñas (p. ej., Cranfield), muchos términos raros o no vistos
        # producen probabilidades cercanas a 0. La suavización de Dirichlet evita ceros y
        # estabiliza la divergencia KL.
        self.mu_bg = float(mu_bg)
        vocab_size = len(index_builder.term_df) if getattr(index_builder, 'term_df', None) else 0
        # Evitar división por cero si el vocabulario no está disponible
        self._uniform_prior = 1.0 / max(1, vocab_size)
        
        # Debug data storage
        self.debug_data = {}

    def compute_scores_batch(self, processed_queries: Dict[str, list]) -> Dict[str, float]:
        """Batch compute clarity scores using retrieval scores"""
        if self.retrieval_results.empty:
            logger.warning("Empty retrieval results - returning zero scores")
            return {qid: 0.0 for qid in processed_queries}

        clarity_scores = {}
        for qid in processed_queries:
            try:
                docs = self.retrieval_results[self.retrieval_results['qid'] == qid]
                clarity_scores[qid] = self.compute_score(docs)
            except Exception as e:
                logger.error(f"Error processing {qid}: {e}", exc_info=True)
                clarity_scores[qid] = 0.0
                clarity_scores[qid] = 0.0
        
        # Save debug data to JSON
        try:
            filename_debug = f'clarity_debug_data_{self.dataset_name}.json' if self.dataset_name else 'clarity_debug_data.json'
            with open(filename_debug, 'w', encoding='utf-8') as f:
                json.dump(self.debug_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved detailed clarity debug data to {filename_debug}")
            
            filename_scores = f'clarity_qpp_scores_{self.dataset_name}.json' if self.dataset_name else 'clarity_qpp_scores.json'
            with open(filename_scores, 'w', encoding='utf-8') as f:
                json.dump(clarity_scores, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved clarity scores to {filename_scores}")
        except Exception as e:
            logger.error(f"Failed to save debug data: {e}")
            
        return clarity_scores

    def compute_score(self, docs: pd.DataFrame) -> float:
        """Core clarity computation for a single query"""
        # Get top documents by retrieval score
        top_docs = docs.nlargest(self.top_k, self.score_column)
        
        if len(top_docs) < 5:
            logger.debug(f"Insufficient docs ({len(top_docs)}) for reliable score")
            return 0.0

        # 1. Build score-weighted term distribution
        term_weights = self._compute_term_weights(top_docs)
        if not term_weights:
            return 0.0

        # 2. Normalize weights using total term weights sum (Fix: ensure P(w|R) sums to 1.0)
        total_weight = sum(term_weights.values())
        if total_weight <= 0:
            return 0.0
            
        p_w_rm = {term: weight/total_weight for term, weight in term_weights.items()}

        # 3. Get collection probabilities
        p_w_coll = self._get_collection_probabilities(p_w_rm.keys())

        # 4. Calculate KL divergence with numerical stability
        kl_divergence = 0.0
        term_contributions = {}
        
        for term, p in p_w_rm.items():
            coll_p = max(p_w_coll.get(term, 1e-10), 1e-10)
            contribution = p * np.log2(p / coll_p)
            kl_divergence += contribution
            term_contributions[term] = {
                "p_w_rm": float(p),
                "p_w_coll": float(coll_p),
                "contribution": float(contribution)
            }

        final_score = max(0.0, kl_divergence)

        # Store debug info for this query
        if not docs.empty:
            qid = str(docs.iloc[0]['qid'])
            self.debug_data[qid] = {
                "score": float(final_score),
                "top_docs_count": len(top_docs),
                "total_weight_sum": float(total_weight),
                "terms": term_contributions
            }

        return final_score

    def _compute_term_weights(self, docs: pd.DataFrame) -> Dict[str, float]:
        """Build term weights using document retrieval scores"""
        term_weights = defaultdict(float)
        
        for doc_score, text in zip(docs[self.score_column], docs['text']):
            if pd.isna(text):
                continue
                
            terms = preprocess_text(text, self.dataset_name)
            for term in terms:
                term_weights[term] += doc_score

        # Apply term cutoff from paper
        sorted_terms = sorted(term_weights.items(), 
                            key=lambda x: x[1], 
                            reverse=True)[:self.term_cutoff]
        
        return dict(sorted_terms)

    def _get_collection_probabilities(self, terms: Iterable[str]) -> Dict[str, float]:
        """Calcula P(w|C) con suavización de Dirichlet.

        Español: Aplicamos Dirichlet al modelo de fondo para reducir varianza en
        colecciones pequeñas y evitar probabilidades cero en términos no vistos.
        Fórmula: (cf + mu * p0) / (total_terms + mu), con p0 uniforme.
        """
        total_terms = max(1, self.index_stats['total_terms'])
        mu = self.mu_bg
        p0 = self._uniform_prior

        probs = {}
        for term in terms:
            cf = float(self.index_stats['term_cf'].get(term, 0))
            probs[term] = (cf + mu * p0) / (total_terms + mu)
        return probs