from metodos.base import PostRetrievalMethod

import numpy as np

import json



class NQC(PostRetrievalMethod):

    def __init__(self, index_builder, retrieval_results, results_df=None):
        super().__init__(index_builder, retrieval_results, results_df)
        
        print(f"Loaded term_df with {len(self.term_df)} terms.")
        print(f"Loaded term_cf with {len(self.term_cf)} terms.")

    

    def _get_term_cf(self, term):
        """
        Gets collection frequency for a term with fallback to lexicon.
        """
        if term in self.term_cf:
            return self.term_cf[term]
        else:
            # Fallback to using the index's lexicon
            lexicon = self.index.getLexicon()
            lex_entry = lexicon.getLexiconEntry(term)
            return lex_entry.getFrequency() if lex_entry is not None else 0

    

    def _init_scores_vec(self, query_id, list_size_param=10):
        relevant_docs = self.results_df[self.results_df['qid'] == query_id].head(list_size_param)
        
        if 'docScore' not in relevant_docs.columns:
            raise KeyError("Column 'docScore' not found in retrieval results.")
        scores = relevant_docs['docScore'].tolist()
        if not scores:
            print("No scores found in the top retrieval results.")
            return np.array([0.0])
        #print(f"Initialized scores_vec: {scores}")
        return np.array(scores)

    

    def _calc_corpus_score(self):
        """
        Calculates the corpus score using collection frequencies.
        """
        if not self.query_terms:
            print("No query terms provided.")
            return 0.0
        cf_vec = np.array([self._get_term_cf(term) for term in self.query_terms])
        if self.total_tokens == 0:
            print("Warning: Total tokens in collection is zero.")
            return 0.0
        # Handle cases where cf_vec might contain zeros to avoid log(0)
        with np.errstate(divide='ignore'):
            log_cf = np.log(np.where(cf_vec == 0, 1, cf_vec) / self.total_tokens)
        # Use mean to mitigate impact of outliers
        corpus_score = log_cf.mean()
        #print(f"Calculated corpus_score: {corpus_score}")
        return corpus_score

    

    def compute_score(self, query_id, query_terms, list_size_param=10):
        """
        Computes the NQC score for a given query.

        Args:
            query_id (str): The ID of the query.
            query_terms (list): Preprocessed terms of the query.
            list_size_param (int): The number of top documents to consider.

        Returns:
            float: The NQC score.
        """
        self.query_terms = query_terms
        if self.results_df is None or self.results_df.empty:
            print(f"No retrieval results available for Query ID: {query_id}")
            return 0.0

        self.scores_vec = self._init_scores_vec(query_id, list_size_param)
        
        # If we got a zero score vector for invalid query, return 0
        if len(self.scores_vec) == 1 and self.scores_vec[0] == 0.0:
            return 0.0
        
        self.ql_corpus_score = self._calc_corpus_score()
        return self.calc_nqc(list_size_param)

    

    def calc_nqc(self, list_size_param):
        """
        Calculates the NQC score following Shtok et al.'s method.
        A. Shtok, O. Kurland, and D. Carmel. Predicting query performance by query-drift estimation

        Args:
            list_size_param (int): The number of top documents to consider.

        Returns:
            float: The NQC score.
        """
        scores_vec = self.scores_vec[:list_size_param]
        if self.ql_corpus_score == 0:
            print("Corpus score is zero; returning NQC score as 0.0 to avoid division by zero.")
            return 0.0

        # Remove epsilon handling since we now catch zero scores earlier
        std_dev = np.std(scores_vec)
        if std_dev == 0.0:
            return 0.0
        
        nqc_score = std_dev / abs(self.ql_corpus_score)
        return nqc_score

    def compute_scores_batch(self, queries_terms_dict, list_size_param=10):
        """
        Computes NQC scores for multiple queries in batch.

        Args:
            queries_terms_dict (dict): Mapping from query_id to list of query terms.
            list_size_param (int): Number of top documents to consider per query.

        Returns:
            dict: Mapping from query_id to its corresponding NQC score.
        """
        scores_dict = {}
        for query_id, query_terms in queries_terms_dict.items():
            scores_dict[query_id] = self.compute_score(query_id, query_terms, list_size_param)
        return scores_dict




