import unittest
import numpy as np
import os
import shutil
import pyterrier as pt
import pandas as pd
from metodos.post_retrieval.uef import UEF
from data.dataset_processor import DatasetProcessor
from indexing.index_builder import IndexBuilder
from utils.text_processing import preprocess_text

class TestUEF(unittest.TestCase):
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
        cls.test_index_path = os.path.join(script_dir, "..", "..", "..", "indices", "test_index_uef")
        
        # Clean up any existing index
        if os.path.exists(cls.test_index_path):
            shutil.rmtree(cls.test_index_path)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(cls.test_index_path), exist_ok=True)
        
        # Build or load index
        cls.index = cls.index_builder.load_or_build_index(cls.test_index_path)
        
        # Create mock retrieval results (simulating BM25 output)
        cls.retrieval_results = pd.DataFrame([
            {"qid": "0", "docno": "doc0", "docScore": 5.2, "rank": 0},
            {"qid": "0", "docno": "doc1", "docScore": 4.8, "rank": 1},
            {"qid": "0", "docno": "doc2", "docScore": 4.1, "rank": 2},
            {"qid": "0", "docno": "doc3", "docScore": 3.5, "rank": 3},
            {"qid": "1", "docno": "doc1", "docScore": 6.1, "rank": 0},
            {"qid": "1", "docno": "doc0", "docScore": 5.3, "rank": 1},
            {"qid": "1", "docno": "doc4", "docScore": 4.2, "rank": 2},
            {"qid": "2", "docno": "doc3", "docScore": 5.8, "rank": 0},
            {"qid": "2", "docno": "doc5", "docScore": 5.1, "rank": 1},
            {"qid": "2", "docno": "doc6", "docScore": 4.5, "rank": 2},
        ])
        
        # Create simulated RM3 results (slightly perturbed scores to simulate re-ranking)
        cls.rm_results = cls.retrieval_results.copy()
        np.random.seed(42)
        cls.rm_results['docScore'] = cls.rm_results['docScore'] + np.random.normal(0, 0.1, len(cls.rm_results))
        
        # Store preprocessed queries for testing
        cls.query_terms = {
            "0": preprocess_text("playa cavancha iquique", dataset_name="iquique_dataset"),
            "1": preprocess_text("zona franca zofri", dataset_name="iquique_dataset"),
            "2": preprocess_text("museo historia iquique", dataset_name="iquique_dataset")
        }
        
        # Create sample predictor scores (simulating WIG/NQC output)
        cls.predictor_scores = {
            "0": 0.75,
            "1": 0.60,
            "2": 0.85
        }

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        try:
            if os.path.exists(cls.test_index_path):
                shutil.rmtree(cls.test_index_path)
        except PermissionError:
            # Windows may lock index files while Java is running
            pass

    def _create_uef(self):
        """Helper to create a UEF instance for testing"""
        return UEF(self.index_builder, self.retrieval_results, self.rm_results, dataset_name="iquique_dataset")

    def test_compute_score_valid_query(self):
        """Test UEF score computation for a valid query"""
        print("\nRunning test_compute_score_valid_query...")
        
        uef = self._create_uef()
        query_id = "0"
        predictor_score = self.predictor_scores[query_id]
        
        score = uef.compute_score(query_id, predictor_score)
        
        # UEF score should be correlation * predictor_score
        self.assertIsInstance(score, float)
        print(f"[OK] Valid query UEF score: {score:.4f}")
        
    def test_compute_score_correlation_bounds(self):
        """Test that UEF maintains correlation bounds"""
        print("\nRunning test_compute_score_correlation_bounds...")
        
        uef = self._create_uef()
        for qid in ["0", "1", "2"]:
            predictor_score = self.predictor_scores[qid]
            uef_score = uef.compute_score(qid, predictor_score)
            
            # If predictor_score != 0, UEF/predictor should give correlation in [-1, 1]
            if predictor_score != 0 and uef_score != 0:
                correlation = uef_score / predictor_score
                self.assertGreaterEqual(correlation, -1.0)
                self.assertLessEqual(correlation, 1.0)
                print(f"[OK] Query {qid}: correlation = {correlation:.4f}")
        
        print("[OK] Correlation bounds verified")

    def test_compute_score_invalid_query(self):
        """Test UEF score computation for an invalid query"""
        print("\nRunning test_compute_score_invalid_query...")
        
        uef = self._create_uef()
        score = uef.compute_score("999", 0.5)
        self.assertEqual(score, 0.0)
        print(f"[OK] Invalid query returns 0.0: {score}")

    def test_compute_score_zero_predictor(self):
        """Test UEF score when base predictor score is zero"""
        print("\nRunning test_compute_score_zero_predictor...")
        
        uef = self._create_uef()
        score = uef.compute_score("0", 0.0)
        self.assertEqual(score, 0.0)
        print(f"[OK] Zero predictor score returns 0.0: {score}")

    def test_compute_scores_batch(self):
        """Test batch computation of UEF scores"""
        print("\nRunning test_compute_scores_batch...")
        
        uef = self._create_uef()
        scores = uef.compute_scores_batch(
            self.query_terms, 
            self.predictor_scores,
            list_size=10
        )
        
        self.assertEqual(len(scores), len(self.query_terms))
        self.assertTrue(all(isinstance(score, float) for score in scores.values()))
        
        print("Batch UEF scores:")
        for qid, score in scores.items():
            print(f"  Query {qid}: {score:.4f}")
        print("[OK] Batch computation successful")

    def test_dataframe_column_cleaning(self):
        """Test that dataframe columns are properly standardized"""
        print("\nRunning test_dataframe_column_cleaning...")
        
        uef = self._create_uef()
        # Verify internal column names are standardized
        self.assertIn('query_id', uef.retrieval_results.columns)
        self.assertIn('doc_id', uef.retrieval_results.columns)
        self.assertIn('score', uef.retrieval_results.columns)
        print("[OK] Column standardization verified")

    def test_high_correlation_scenario(self):
        """Test UEF with identical rankings (perfect correlation)"""
        print("\nRunning test_high_correlation_scenario...")
        
        # Create identical results for perfect correlation
        identical_rm = self.retrieval_results.copy()
        uef_perfect = UEF(self.index_builder, self.retrieval_results, identical_rm)
        
        score = uef_perfect.compute_score("0", 1.0)
        
        # With identical rankings, correlation should be 1.0, so UEF = 1.0 * predictor
        # Due to floating point, check if close to 1.0
        self.assertAlmostEqual(score, 1.0, places=5)
        print(f"[OK] Perfect correlation score: {score:.4f}")

    def test_negative_predictor_score(self):
        """Test UEF with negative predictor score"""
        print("\nRunning test_negative_predictor_score...")
        
        uef = self._create_uef()
        score = uef.compute_score("0", -0.5)
        
        # UEF should handle negative predictor scores
        self.assertIsInstance(score, float)
        print(f"[OK] Negative predictor score handled: {score:.4f}")

if __name__ == '__main__':
    unittest.main()
