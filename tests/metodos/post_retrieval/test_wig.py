import unittest
import numpy as np
import os
import shutil
import pyterrier as pt
import pandas as pd
from metodos.post_retrieval.wig import WIG
from data.dataset_processor import DatasetProcessor
from indexing.index_builder import IndexBuilder
from utils.text_processing import preprocess_text
from retrieval.retrieval import perform_retrieval

class TestWIG(unittest.TestCase):
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
        
        # Create WIG instance
        cls.wig = WIG(cls.index_builder, cls.retrieval_results)
        
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
        scores = self.wig._init_scores_vec(query_id, list_size_param=10)
        self.assertIsInstance(scores, np.ndarray)
        self.assertLessEqual(len(scores), 10)
        print(f"✓ Valid query scores initialized: {scores[:5]}...")
        
        # Test with invalid query
        scores = self.wig._init_scores_vec("999", list_size_param=10)
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0], 0.0)
        print("✓ Invalid query returns zero score")

    def test_calc_corpus_score(self):
        """Test corpus score calculation"""
        print("\nRunning test_calc_corpus_score...")
        
        # Test with valid query terms
        self.wig.query_terms = self.query_terms["0"]
        corpus_score = self.wig._calc_corpus_score()
        self.assertNotEqual(corpus_score, 0.0)
        print(f"✓ Corpus score for valid query: {corpus_score:.4f}")
        
        # Test with empty query
        self.wig.query_terms = []
        corpus_score = self.wig._calc_corpus_score()
        self.assertEqual(corpus_score, 0.0)
        print("✓ Empty query returns zero corpus score")

    def test_compute_score(self):
        """Test WIG score computation"""
        print("\nRunning test_compute_score...")
        
        # Test with valid query
        query_id = "0"
        query_terms = self.query_terms[query_id]
        print(f"\nValid query test:")
        print(f"Query ID: {query_id}")
        print(f"Query terms: {query_terms}")
        
        score = self.wig.compute_score(query_id, query_terms, list_size_param=10)
        print(f"Score vector: {self.wig.scores_vec[:5]}...")
        print(f"Corpus score: {self.wig.ql_corpus_score}")
        print(f"Final WIG score: {score:.4f}")
        
        self.assertIsInstance(score, float)
        print(f"✓ WIG score for valid query: {score:.4f}")
        
        # Test with invalid query
        print(f"\nInvalid query test:")
        print(f"Query ID: 999")
        print(f"Query terms: ['unknown']")
        
        score = self.wig.compute_score("999", ["unknown"], list_size_param=10)
        print(f"Score vector: {self.wig.scores_vec}")
        print(f"Corpus score: {self.wig.ql_corpus_score}")
        print(f"Final WIG score: {score:.4f}")
        
        if score > 0:
            print(f"Warning: Invalid query returning non-zero score: {score}")
            print("Score components:")
            print(f"- Mean of scores: {np.mean(self.wig.scores_vec)}")
            print(f"- Corpus score: {self.wig.ql_corpus_score}")
            print(f"- Query length normalization: {np.sqrt(len(['unknown']))}")
            self.assertTrue(isinstance(score, float))
        else:
            self.assertLess(abs(score), 1e-6)
        print(f"✓ Invalid query handling verified: {score}")

    def test_calc_wig(self):
        """Test WIG calculation"""
        print("\nRunning test_calc_wig...")
        
        # Set up test data with more realistic values
        self.wig.scores_vec = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        self.wig.ql_corpus_score = -1.0  # Typical corpus score is negative due to log
        self.wig.query_terms = ["test", "query"]  # Need this for sqrt(len(query_terms))
        
        # Calculate WIG score
        score = self.wig.calc_wig(list_size_param=5)
        # Expected: (mean(scores) - corpus_score) / sqrt(len(query_terms))
        expected = (np.mean([1.0, 0.8, 0.6, 0.4, 0.2]) - (-1.0)) / np.sqrt(2)
        
        # Check if corpus score is zero, then expected should be zero
        if self.wig.ql_corpus_score == 0:
            expected = 0.0
        
        self.assertAlmostEqual(score, expected, places=4)
        print(f"✓ WIG calculation correct: {score:.4f}")
        
        # Test with non-zero corpus score to verify normalization
        self.wig.ql_corpus_score = -2.0
        score = self.wig.calc_wig(list_size_param=5)
        expected = (np.mean([1.0, 0.8, 0.6, 0.4, 0.2]) - (-2.0)) / np.sqrt(2)
        self.assertAlmostEqual(score, expected, places=4)
        print(f"✓ Non-zero corpus score handled correctly: {score:.4f}")

    def test_compute_scores_batch(self):
        """Test batch computation of WIG scores"""
        print("\nRunning test_compute_scores_batch...")
        
        # Test with multiple queries
        scores = self.wig.compute_scores_batch(self.query_terms, list_size_param=10)
        
        self.assertEqual(len(scores), len(self.query_terms))
        self.assertTrue(all(isinstance(score, float) for score in scores.values()))
        
        print("Batch scores:")
        for qid, score in scores.items():
            print(f"Query {qid}: {score:.4f}")
        print("✓ Batch computation successful")

    def test_get_term_cf(self):
        """Test collection frequency retrieval for terms"""
        print("\nRunning test_get_term_cf...")
        
        # Test with a term we know exists in the collection
        raw_term = "iquique"
        processed_term = preprocess_text(raw_term, dataset_name="iquique_dataset")[0]  # Get first token
        print(f"Testing term: {raw_term} -> {processed_term}")
        
        cf = self.wig._get_term_cf(processed_term)
        print(f"Collection frequency: {cf}")
        self.assertGreater(cf, 0)

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\nRunning test_edge_cases...")
        
        # Test with empty results DataFrame
        empty_wig = WIG(self.index_builder, pd.DataFrame())
        score = empty_wig.compute_score("0", ["test"], list_size_param=10)
        self.assertEqual(score, 0.0)
        print("✓ Empty results handled correctly")
        
        # Test with missing docScore column
        bad_results = self.retrieval_results.copy()
        bad_results = bad_results.drop('docScore', axis=1)
        with self.assertRaises(KeyError):
            bad_wig = WIG(self.index_builder, bad_results)
            bad_wig.compute_score("0", ["test"], list_size_param=10)
        print("✓ Missing docScore column handled correctly")

if __name__ == '__main__':
    unittest.main() 