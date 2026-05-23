from metodos.base import PreRetrievalMethod
import numpy as np
import json

class IDF(PreRetrievalMethod):
    def __init__(self, index):
        super().__init__(index)
        if hasattr(index, 'term_df'):
            # If we're passed an IndexBuilder, use its statistics
            self.term_df = index.term_df
            self.total_docs = index.total_docs
            self.index_builder = index
            self.index = index.index  # Get PyTerrier index for fallback
        else:
            # Otherwise, load from PyTerrier index
            self.index_builder = None
            self.index = index
            self.total_docs = index.getCollectionStatistics().getNumberOfDocuments()
            # Avoid iterating the entire lexicon on large corpora; fetch on-demand
            self.term_df = {}

    def compute_score(self, query_terms, method='avg', **kwargs):
        """
        Compute IDF score for a query with improved OOV handling.

        Args:
            query_terms (list): List of preprocessed query terms
            method (str): Aggregation method ('avg' or 'max')

        Returns:
            float: IDF score
        """
        if not query_terms:
            return 0.0

        idfs = []
        for term in query_terms:
            df = self._get_term_df(term)

            # Handle OOV terms (df == -1)
            if df == -1:
                if method == 'max':
                    # For max method, treat OOV as idf=0 (or skip entirely)
                    continue  # Skip OOV terms in max calculation
                else:  # 'avg' method
                    continue  # Skip OOV terms in average calculation

            # Apply add-1 smoothing to avoid division by zero for in-vocabulary terms
            df += 1
            idf = np.log((self.total_docs + 1) / df) 
            idfs.append(idf)

        if not idfs:
            return 0.0

        if method == 'max':
            return max(0.0, max(idfs))
        elif method == 'avg':
            return max(0.0, np.mean(idfs))
        else:
            raise ValueError("Invalid method. Choose 'max' or 'avg'.")

    def _get_term_df(self, term):
        """Get document frequency for a term.

        Returns:
            int: Document frequency if term exists, -1 if OOV (out of vocabulary)
        """
        # First try our tracked statistics
        if term in self.term_df:
            return self.term_df[term]

        # If we have an index builder, try its statistics
        if self.index_builder and term in self.index_builder.term_df:
            return self.index_builder.term_df[term]

        # Fallback to lexicon lookup
        lexicon = self.index.getLexicon()
        lex_entry = lexicon.getLexiconEntry(term)
        if lex_entry is not None:
            # PyTerrier/Terrier API compatibility: entry may be LexiconEntry or Map.Entry
            if hasattr(lex_entry, 'getDocumentFrequency'):
                df = lex_entry.getDocumentFrequency()
            elif hasattr(lex_entry, 'getValue') and hasattr(lex_entry.getValue(), 'getDocumentFrequency'):
                df = lex_entry.getValue().getDocumentFrequency()
            else:
                df = 0
            # Cache the result
            self.term_df[term] = df
            return df

        # Return -1 for OOV terms (instead of 0) to distinguish from terms with df=0
        return -1

    def compute_scores_batch(self, queries_dict=None, method='avg'):
        """
        Compute IDF scores for multiple queries.
        
        Args:
            queries_dict (dict): Mapping from query_id to preprocessed query terms
            method (str): Aggregation method ('max' or 'avg')
            
        Returns:
            dict: Mapping from query_id to IDF score
        """
        scores_dict = {}
        for query_id, query_terms in queries_dict.items():
            scores_dict[query_id] = self.compute_score(query_terms, method=method)
        return scores_dict
