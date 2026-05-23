import unittest
import pandas as pd
import numpy as np
from evaluation.evaluator import evaluate_results
from data.iquique_dataset import IquiqueDataset

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        """Set up test data using IquiqueDataset"""
        self.dataset = IquiqueDataset()
        self.qrels = self.dataset.get_qrels()
    
        
        # Create a sample run dataframe that matches exactly with IquiqueDataset qrels
        self.perfect_run = pd.DataFrame([
            # Query 0 - Perfect ranking for "playa cavancha iquique"
            {"qid": "0", "doc_id": "doc2", "docScore": 1.0},  # Relevant (1)
            {"qid": "0", "doc_id": "doc0", "docScore": 0.5},  # Not relevant
            {"qid": "0", "doc_id": "doc1", "docScore": 0.3},  # Not relevant
            
            # Query 1 - Perfect ranking for "zona franca zofri"
            {"qid": "1", "doc_id": "doc1", "docScore": 1.0},  # Relevant (1)
            {"qid": "1", "doc_id": "doc0", "docScore": 0.5},  # Not relevant
            
            # Query 2 - Perfect ranking for "museo historia iquique"
            {"qid": "2", "doc_id": "doc3", "docScore": 1.0},  # Relevant (1)
            {"qid": "2", "doc_id": "doc0", "docScore": 0.9},  # Relevant (1)
            {"qid": "2", "doc_id": "doc1", "docScore": 0.5},  # Not relevant
            
            # Query 3 - Perfect ranking for "historia salitre guerra pacifico"
            {"qid": "3", "doc_id": "doc6", "docScore": 1.0},  # Highly relevant (2)
            {"qid": "3", "doc_id": "doc5", "docScore": 0.9},  # Relevant (1)
            {"qid": "3", "doc_id": "doc3", "docScore": 0.8},  # Relevant (1)
            {"qid": "3", "doc_id": "doc7", "docScore": 0.7},  # Relevant (1)
        ])
        
        # Print run to debug
        print("\nPerfect run data:")
        print(self.perfect_run)
        
        # Create a reversed (bad) ranking for comparison
        self.reversed_run = pd.DataFrame([
            # Query 0 - Reversed ranking
            {"qid": "0", "doc_id": "doc1", "docScore": 1.0},  # Not relevant
            {"qid": "0", "doc_id": "doc0", "docScore": 0.8},  # Not relevant
            {"qid": "0", "doc_id": "doc2", "docScore": 0.5},  # Relevant (at bottom)
            
            # Query 1 - Reversed ranking
            {"qid": "1", "doc_id": "doc0", "docScore": 1.0},  # Not relevant
            {"qid": "1", "doc_id": "doc1", "docScore": 0.5},  # Relevant (at bottom)
            
            # Query 2 - Reversed ranking
            {"qid": "2", "doc_id": "doc1", "docScore": 1.0},  # Not relevant
            {"qid": "2", "doc_id": "doc3", "docScore": 0.5},  # Relevant (at bottom)
            {"qid": "2", "doc_id": "doc0", "docScore": 0.3},  # Relevant (at bottom)
            
            # Query 3 - Reversed ranking
            {"qid": "3", "doc_id": "doc4", "docScore": 1.0},  # Not relevant
            {"qid": "3", "doc_id": "doc2", "docScore": 0.9},  # Not relevant
            {"qid": "3", "doc_id": "doc6", "docScore": 0.5},  # Highly relevant (at bottom)
            {"qid": "3", "doc_id": "doc5", "docScore": 0.4},  # Relevant (at bottom)
        ])

    def test_perfect_ndcg(self):
        """Test NDCG@10 for perfect ranking"""
        print("\nRunning test_perfect_ndcg...")
        
        results = evaluate_results(
            self.qrels,
            self.perfect_run,
            metrics=['ndcg@10'],
            dataset_name="iquique_dataset",
            min_results=1
        )
        
        ndcg_score = results['ndcg@10']['mean']
        print(f"NDCG@10 Score for perfect ranking: {ndcg_score:.4f}")
        print("Per-query scores:", results['ndcg@10']['per_query'])
        
        self.assertGreater(ndcg_score, 0.8)
        print("✓ Perfect NDCG@10 test passed")

    def test_reversed_ndcg(self):
        """Test NDCG@10 for reversed (worst) ranking"""
        print("\nRunning test_reversed_ndcg...")
        
        results = evaluate_results(
            self.qrels,
            self.reversed_run,
            metrics=['ndcg@10'],
            dataset_name="iquique_dataset",
            min_results=1
        )
        
        ndcg_score = results['ndcg@10']['mean']
        print(f"NDCG@10 Score for reversed ranking: {ndcg_score:.4f}")
        print("Per-query scores:", results['ndcg@10']['per_query'])
        
        self.assertLess(ndcg_score, 0.6)
        print("✓ Reversed NDCG@10 test passed")

    def test_perfect_ap(self):
        """Test AP for perfect ranking"""
        print("\nRunning test_perfect_ap...")
        
        results = evaluate_results(
            self.qrels,
            self.perfect_run,
            metrics=['ap'],
            dataset_name="iquique_dataset",
            min_results=1
        )
        
        ap_score = results['ap']['mean']
        print(f"AP Score for perfect ranking: {ap_score:.4f}")
        print("Per-query scores:", results['ap']['per_query'])
        
        self.assertGreater(ap_score, 0.7)
        print("✓ Perfect AP test passed")

    def test_reversed_ap(self):
        """Test AP for reversed ranking"""
        print("\nRunning test_reversed_ap...")
        
        results = evaluate_results(
            self.qrels,
            self.reversed_run,
            metrics=['ap'],
            dataset_name="iquique_dataset",
            min_results=1
        )
        
        ap_score = results['ap']['mean']
        print(f"AP Score for reversed ranking: {ap_score:.4f}")
        print("Per-query scores:", results['ap']['per_query'])
        
        self.assertLess(ap_score, 0.5)
        print("✓ Reversed AP test passed")

    def test_multiple_metrics(self):
        """Test multiple metrics at once"""
        print("\nRunning test_multiple_metrics...")
        
        metrics = ['ndcg@10', 'ndcg@20', 'ap']
        results = evaluate_results(
            self.qrels,
            self.perfect_run,
            metrics=metrics,
            dataset_name="iquique_dataset",
            min_results=1
        )
        
        for metric in metrics:
            print(f"\n{metric} results:")
            print(f"Mean score: {results[metric]['mean']:.4f}")
            print("Per-query scores:", results[metric]['per_query'])
            
        print("✓ Multiple metrics test passed")

    def test_invalid_queries(self):
        """Test handling of queries not in qrels"""
        print("\nRunning test_invalid_queries...")
        
        invalid_run = pd.DataFrame([
            {"qid": "999", "doc_id": "doc1", "docScore": 1.0}
        ])
        
        results = evaluate_results(
            self.qrels,
            invalid_run,
            metrics=['ndcg@10', 'ap'],
            dataset_name="iquique_dataset"
        )
        
        print("Results for invalid queries:")
        print(f"NDCG@10 mean: {results['ndcg@10']['mean']}")
        print(f"AP mean: {results['ap']['mean']}")
        print("Per-query scores:", results['ndcg@10']['per_query'])
        
        self.assertEqual(results['ndcg@10']['mean'], 0.0)
        self.assertEqual(results['ap']['mean'], 0.0)
        print("✓ Invalid queries test passed")

if __name__ == '__main__':
    unittest.main() 