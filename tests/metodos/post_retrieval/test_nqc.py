import unittest
import numpy as np
import os
import shutil
import pyterrier as pt
import pandas as pd
from metodos.post_retrieval.nqc import NQC
from data.dataset_processor import DatasetProcessor
from indexing.index_builder import IndexBuilder
from utils.text_processing import preprocess_text
from retrieval.retrieval import perform_retrieval

class TestNQC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that can be reused for all tests"""
        # Initialize PyTerrier if not already started
        if not pt.started():
            pt.init()
            
        # Create dataset processor with IquiqueDataset
        cls.dataset_processor = DatasetProcessor("iquique_dataset")
        cls.index_builder = IndexBuilder(cls.dataset_processor, "iquique_test")
        
        # Create a temporary index path for testing
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_index_path = os.path.join(script_dir, "..", "..", "..", "indices", "test_index")
        
        # Clean up any existing index
        try:
            if os.path.exists(cls.test_index_path):
                shutil.rmtree(cls.test_index_path)
        except PermissionError:
            pass
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(cls.test_index_path), exist_ok=True)
        
        # Build or load index
        cls.index = cls.index_builder.load_or_build_index(cls.test_index_path)
        
        # Create sample queries
        cls.queries_df = pd.DataFrame([
            {"qid": "0", "query": "playa cavancha iquique"},
            {"qid": "1", "query": "zona franca zofri"},
            {"qid": "2", "query": "museo historia iquique"}
        ])
        
        
        # Create mock retrieval results
        cls.retrieval_results = pd.DataFrame([
            {"qid": "0", "docno": "doc2", "docScore": 5.2, "rank": 0},
            {"qid": "0", "docno": "doc0", "docScore": 4.8, "rank": 1},
            {"qid": "0", "docno": "doc1", "docScore": 4.1, "rank": 2},
            {"qid": "1", "docno": "doc1", "docScore": 6.1, "rank": 0},
            {"qid": "1", "docno": "doc0", "docScore": 5.3, "rank": 1},
            {"qid": "2", "docno": "doc3", "docScore": 5.8, "rank": 0},
            {"qid": "2", "docno": "doc0", "docScore": 5.1, "rank": 1},
        ])
        
        # Create NQC instance
        cls.nqc = NQC(cls.index_builder, cls.retrieval_results)
        
        # Store preprocessed queries for testing
        cls.query_terms = {
            qid: preprocess_text(query)
            for qid, query in cls.queries_df.set_index('qid')['query'].items()
        }

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        try:
            if os.path.exists(cls.test_index_path):
                shutil.rmtree(cls.test_index_path)
        except PermissionError:
            pass

    def test_init_scores_vec(self):
        """Test initialization of scores vector"""
        print("\nRunning test_init_scores_vec...")
        
        # Test with valid query
        query_id = "0"
        scores = self.nqc._init_scores_vec(query_id, list_size_param=10)
        self.assertIsInstance(scores, np.ndarray)
        self.assertLessEqual(len(scores), 10)
        print(f"✓ Valid query scores initialized: {scores[:5]}...")
        
        # Test with invalid query
        scores = self.nqc._init_scores_vec("999", list_size_param=10)
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0], 0.0)
        print("✓ Invalid query returns zero score")

    def test_calc_corpus_score(self):
        """Test corpus score calculation"""
        print("\nRunning test_calc_corpus_score...")
        
        # Test with valid query terms
        self.nqc.query_terms = self.query_terms["0"]
        corpus_score = self.nqc._calc_corpus_score()
        self.assertNotEqual(corpus_score, 0.0)
        print(f"✓ Corpus score for valid query: {corpus_score:.4f}")
        
        # Test with empty query
        self.nqc.query_terms = []
        corpus_score = self.nqc._calc_corpus_score()
        self.assertEqual(corpus_score, 0.0)
        print("✓ Empty query returns zero corpus score")

    def test_compute_score(self):
        """Test NQC score computation"""
        print("\nRunning test_compute_score...")
        
        # Test with valid query
        query_id = "0"
        query_terms = preprocess_text("playa cavancha iquique", dataset_name="iquique_dataset")
        print(f"\nValid query test:")
        print(f"Query ID: {query_id}")
        print(f"Query terms: {query_terms}")
        
        score = self.nqc.compute_score(query_id, query_terms, list_size_param=10)
        print(f"Score vector: {self.nqc.scores_vec[:5]}...")
        print(f"Corpus score: {self.nqc.ql_corpus_score}")
        print(f"Final NQC score: {score:.4f}")
        
        self.assertGreater(score, 0.0)
        print(f"✓ NQC score for valid query: {score:.4f}")
        
        # Test with invalid query
        print(f"\nInvalid query test:")
        print(f"Query ID: 999")
        print(f"Query terms: ['unknown']")
        
        score = self.nqc.compute_score("999", ["unknown"], list_size_param=10)
        print(f"Score vector: {self.nqc.scores_vec}")
        print(f"Corpus score: {self.nqc.ql_corpus_score}")
        print(f"Final NQC score: {score:.4f}")
        
        if score > 0:
            print(f"Warning: Invalid query returning non-zero score: {score}")
            print("Score components:")
            print(f"- Standard deviation of scores: {np.std(self.nqc.scores_vec)}")
            print(f"- Corpus score: {self.nqc.ql_corpus_score}")
        
        self.assertLess(score, 1e-6)
        print(f"✓ Invalid query handling verified: {score}")

    def test_calc_nqc(self):
        """Test NQC calculation"""
        print("\nRunning test_calc_nqc...")
        
        # Set up test data
        self.nqc.scores_vec = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        self.nqc.ql_corpus_score = 1.0
        
        # Calculate NQC score
        score = self.nqc.calc_nqc(list_size_param=5)
        expected = np.std([1.0, 0.8, 0.6, 0.4, 0.2])
        self.assertAlmostEqual(score, expected, places=4)
        print(f"✓ NQC calculation correct: {score:.4f}")
        
        # Test with zero corpus score
        self.nqc.ql_corpus_score = 0.0
        score = self.nqc.calc_nqc(list_size_param=5)
        self.assertEqual(score, 0.0)
        print("✓ Zero corpus score handled correctly")

    def test_compute_scores_batch(self):
        """Test batch computation of NQC scores"""
        print("\nRunning test_compute_scores_batch...")
        
        # Test with multiple queries
        scores = self.nqc.compute_scores_batch(self.query_terms, list_size_param=10)
        
        self.assertEqual(len(scores), len(self.query_terms))
        self.assertTrue(all(isinstance(score, float) for score in scores.values()))
        
        print("Batch scores:")
        for qid, score in scores.items():
            print(f"Query {qid}: {score:.4f}")
        print("✓ Batch computation successful")

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\nRunning test_edge_cases...")
        
        # Test with empty results DataFrame
        empty_nqc = NQC(self.index_builder, pd.DataFrame())
        score = empty_nqc.compute_score("0", ["test"], list_size_param=10)
        self.assertEqual(score, 0.0)
        print("✓ Empty results handled correctly")
        
        # Test with missing docScore column
        bad_results = self.retrieval_results.copy()
        bad_results = bad_results.drop('docScore', axis=1)
        with self.assertRaises(KeyError):
            bad_nqc = NQC(self.index_builder, bad_results)
            bad_nqc.compute_score("0", ["test"], list_size_param=10)
        print("✓ Missing docScore column handled correctly")

if __name__ == '__main__':
    unittest.main() 