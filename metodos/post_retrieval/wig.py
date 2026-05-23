from metodos.base import PostRetrievalMethod

import numpy as np
import json


class WIG(PostRetrievalMethod):

    def __init__(self, index_builder, retrieval_results, results_df=None):
        super().__init__(index_builder, retrieval_results, results_df)

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

    def _init_scores_vec(self, query_id,list_size_param=10):
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
        Calculates the BM25 corpus score instead of QL score
        """
        if not self.query_terms:
            print("No query terms provided.")
            return 0.0
        
        # Use average document length for corpus score calculation
        avg_doc_len = self.total_tokens / self.total_docs
        k1 = 1.2  # BM25 k1 parameter
        b = 0.75  # BM25 b parameter
        
        scores = []
        for term in self.query_terms:
            cf = self._get_term_cf(term)
            idf = np.log((self.total_docs - cf + 0.5) / (cf + 0.5) + 1.0)
            tf = cf / self.total_docs  # Average term frequency across collection
            
            # BM25 term score calculation
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (avg_doc_len / avg_doc_len))  # Simplified as ratio is 1
            term_score = idf * (numerator / denominator)
            scores.append(term_score)
        
        return np.mean(scores) if scores else 0.0

    def compute_score(self, query_id, query_terms, list_size_param=10):
        """
        Computes the WIG score for a given query.

        Args:
            query_id (str): The ID of the query.
            query_terms (list): Preprocessed terms of the query.
            list_size_param (int): The number of top documents to consider.

        Returns:
            float: The WIG score.
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
        return self.calc_wig(list_size_param)

    def calc_wig(self, list_size_param):
        """
        Calculates the WIG score following Zhou and Croft's method.
        Y. Zhou and W. B. Croft. Query performance prediction in web search environments

        Args:
            list_size_param (int): The number of top documents to consider.

        Returns:
            float: The WIG score.
        """
        scores_vec = self.scores_vec[:list_size_param]
        if self.ql_corpus_score == 0:
            print("Corpus score is zero; returning WIG score as 0.0 to avoid division by zero.")
            return 0.0
        wig_score = (scores_vec.mean() - self.ql_corpus_score) / np.sqrt(len(self.query_terms))
        return wig_score

    def compute_scores_batch(self, queries_terms_dict, list_size_param=10):
        """
        Computes WIG scores for multiple queries in batch.

        Args:
            queries_terms_dict (dict): Mapping from query_id to list of query terms.
            list_size_param (int): Number of top documents to consider per query.

        Returns:
            dict: Mapping from query_id to its corresponding WIG score.
        """
        scores_dict = {}
        for query_id, query_terms in queries_terms_dict.items():
            scores_dict[query_id] = self.compute_score(query_id, query_terms, list_size_param)
        return scores_dict
