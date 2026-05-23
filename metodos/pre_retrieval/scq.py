from metodos.base import PreRetrievalMethod
import numpy as np
import json

class SCQ(PreRetrievalMethod):
    def __init__(self, index):
        super().__init__(index)
        if hasattr(index, 'term_df'):
            # If we're passed an IndexBuilder, use its statistics
            self.term_df = index.term_df
            self.term_cf = index.term_cf
            self.total_docs = index.total_docs
            self.total_terms = index.total_terms
            self.index_builder = index
            self.index = index.index  # Get PyTerrier index for fallback
        else:
            # Otherwise, load from PyTerrier index
            self.index_builder = None
            self.index = index
            self.total_docs = index.getCollectionStatistics().getNumberOfDocuments()
            self.total_terms = index.getCollectionStatistics().getNumberOfTokens()
            # Avoid iterating the full lexicon; populate on-demand
            self.term_df = {}
            self.term_cf = {}

    def _get_term_stats(self, term):
        """Get document frequency and collection frequency for a term."""
        # First try our tracked statistics
        if term in self.term_df and term in self.term_cf:
            return self.term_df[term], self.term_cf[term]
        
        # If we have an index builder, try its statistics
        if self.index_builder:
            has_df = term in getattr(self.index_builder, 'term_df', {})
            has_cf = term in getattr(self.index_builder, 'term_cf', {})
            if has_df and has_cf:
                return (self.index_builder.term_df[term],
                        self.index_builder.term_cf[term])
        
        # Fallback to lexicon lookup
        lexicon = self.index.getLexicon()
        lex_entry = lexicon.getLexiconEntry(term)
        if lex_entry is not None:
            # Handle both LexiconEntry and Map.Entry wrappers
            if hasattr(lex_entry, 'getDocumentFrequency') and hasattr(lex_entry, 'getFrequency'):
                df = lex_entry.getDocumentFrequency()
                cf = lex_entry.getFrequency()
            elif hasattr(lex_entry, 'getValue'):
                val = lex_entry.getValue()
                df = val.getDocumentFrequency() if hasattr(val, 'getDocumentFrequency') else 0
                cf = val.getFrequency() if hasattr(val, 'getFrequency') else 0
            else:
                df, cf = 0, 0
            # Cache the results
            self.term_df[term] = df
            self.term_cf[term] = cf
            return df, cf
        return 0, 0

    def compute_score(self, query_terms, method='avg', **kwargs):
        """
        Computes SCQ score for a query using specified method.
        
        Args:
            query_terms (list): List of preprocessed query terms
            method (str): The SCQ calculation method ('max', 'avg', or 'sum')
            
        Returns:
            float: The SCQ score
        """
        raw_scq_scores = self._calc_raw_scq(query_terms)
        
        if method == 'max':
            return max(0.0, self.calc_max_scq(raw_scq_scores))
        elif method == 'avg':
            return max(0.0, self.calc_avg_scq(raw_scq_scores))
        elif method == 'sum':
            return max(0.0, self.calc_scq(raw_scq_scores))
        else:
            raise ValueError("Invalid method. Choose 'max', 'avg', or 'sum'.")

    def _calc_raw_scq(self, terms):
        """
        Calculates raw SCQ scores for terms.
        
        Zhao, Y. et al. 2008.
        Effective Pre-retrieval Query Performance Prediction Using Similarity and Variability Evidence
        """
        raw_scores = []
        
        for term in terms:
            df, cf = self._get_term_stats(term)
            if cf > 0 and df > 0:
                score = (1 + np.log(cf)) * np.log(1 + self.total_docs / df)
                raw_scores.append(score)
            else:
                raw_scores.append(0)
                
        return np.array(raw_scores)

    def calc_scq(self, raw_scores):
        """Calculates sum SCQ score."""
        return np.sum(raw_scores)

    def calc_max_scq(self, raw_scores):
        """Calculates maximum SCQ score."""
        return np.max(raw_scores) if len(raw_scores) > 0 else 0.0

    def calc_avg_scq(self, raw_scores):
        """Calculates average SCQ score (NSCQ in the original paper)."""
        return np.mean(raw_scores) if len(raw_scores) > 0 else 0.0

    def compute_scores_batch(self, queries_dict=None, method='avg'):
        """
        Computes SCQ scores for multiple queries in batch.

        Args:
            queries_dict (dict): Mapping from query_id to preprocessed query terms
            method (str): The SCQ calculation method ('max', 'avg', or 'sum')

        Returns:
            dict: Mapping from query_id to its corresponding SCQ score
        """
        scores_dict = {}
        for query_id, query_terms in queries_dict.items():
            scores_dict[query_id] = self.compute_score(query_terms, method=method)
        return scores_dict