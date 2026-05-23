from abc import ABC, abstractmethod

class QPPMethod(ABC):
    def __init__(self, index):
        self.index = index

    @abstractmethod
    def compute_score(self, query, **kwargs):
        pass

class PreRetrievalMethod(QPPMethod):
    pass

class PostRetrievalMethod(QPPMethod):
    def __init__(self, index_builder, retrieval_results, results_df=None):
        self.index = index_builder.index
        self.retrieval_results = retrieval_results
        self.results_df = results_df if results_df is not None else retrieval_results
        self.total_docs = index_builder.total_docs
        self.total_tokens = index_builder.total_terms
        self.term_df = index_builder.term_df
        self.term_cf = index_builder.term_cf

    @abstractmethod
    def compute_score(self, query, results, **kwargs):
        pass
