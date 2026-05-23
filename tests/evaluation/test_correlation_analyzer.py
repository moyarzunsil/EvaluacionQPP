import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from evaluation.correlation_analyzer import QPPCorrelationAnalyzer
from data.iquique_dataset import IquiqueDataset

class TestQPPCorrelationAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that can be reused for all tests"""
        # Create sample QPP scores
        cls.qpp_scores = {
            "q1": {
                "idf_avg": 0.8,
                "idf_max": 0.9,
                "scq_avg": 0.7,
                "scq_max": 0.85,
                "wig": 0.75,
                "nqc": 0.65,
                "clarity": 0.6
            },
            "q2": {
                "idf_avg": 0.6,
                "idf_max": 0.7,
                "scq_avg": 0.5,
                "scq_max": 0.65,
                "wig": 0.55,
                "nqc": 0.45,
                "clarity": 0.4
            },
            "q3": {
                "idf_avg": 0.4,
                "idf_max": 0.5,
                "scq_avg": 0.3,
                "scq_max": 0.45,
                "wig": 0.35,
                "nqc": 0.25,
                "clarity": 0.2
            }
        }

        # Create sample retrieval metrics
        cls.retrieval_metrics = {
            "ndcg@10": {
                "per_query": {
                    "q1": 0.9,
                    "q2": 0.7,
                    "q3": 0.5
                },
                "mean": 0.7
            },
            "ap": {
                "per_query": {
                    "q1": 0.85,
                    "q2": 0.65,
                    "q3": 0.45
                },
                "mean": 0.65
            }
        }

        # Create temporary directory for output
        cls.temp_dir = tempfile.mkdtemp()
        
        # Initialize analyzer
        cls.analyzer = QPPCorrelationAnalyzer(
            cls.qpp_scores,
            cls.retrieval_metrics,
            output_dir=cls.temp_dir
        )

    def test_initialization(self):
        """Test proper initialization of QPPCorrelationAnalyzer"""
        print("\nRunning test_initialization...")
        
        # Check if DataFrames are properly created
        self.assertIsInstance(self.analyzer.qpp_df, pd.DataFrame)
        self.assertIsInstance(self.analyzer.metrics_df, pd.DataFrame)
        
        # Check if dimensions match
        self.assertEqual(len(self.analyzer.qpp_df), len(self.qpp_scores))
        self.assertEqual(len(self.analyzer.metrics_df.columns), len(self.retrieval_metrics))
        
        print("✓ Analyzer initialized correctly")

    def test_calculate_correlations(self):
        """Test correlation calculation methods"""
        print("\nRunning test_calculate_correlations...")
        
        # Test different correlation types
        correlation_types = ['pearson', 'spearman', 'kendall']
        correlations = self.analyzer.calculate_correlations(correlation_types)
        
        for corr_type in correlation_types:
            self.assertIn(corr_type, correlations)
            self.assertIsInstance(correlations[corr_type], pd.DataFrame)
            print(f"\n{corr_type.capitalize()} correlations:")
            print(correlations[corr_type])
            
            # Check correlation values are between -1 and 1
            self.assertTrue((correlations[corr_type].values <= 1).all())
            self.assertTrue((correlations[corr_type].values >= -1).all())
        
        print("✓ Correlations calculated correctly")

    def test_plot_correlation_heatmap(self):
        """Test correlation heatmap plotting"""
        print("\nRunning test_plot_correlation_heatmap...")
        
        # Test plotting with different correlation types
        for corr_type in ['kendall', 'spearman', 'pearson']:
            # Should create a file in temp directory
            self.analyzer.plot_correlation_heatmap(correlation_type=corr_type, save_plot=True)
            expected_file = os.path.join(self.temp_dir, f'qpp_correlation_{corr_type}.pdf')
            self.assertTrue(os.path.exists(expected_file))
            
        print("✓ Correlation heatmaps generated successfully")

    def test_plot_scatter_plots(self):
        """Test scatter plot generation"""
        print("\nRunning test_plot_scatter_plots...")
        
        # Test plotting for each metric
        for metric in self.retrieval_metrics.keys():
            self.analyzer.plot_scatter_plots(metric, save_plots=True)
            expected_file = os.path.join(self.temp_dir, f'qpp_scatter_{metric}.pdf')
            self.assertTrue(os.path.exists(expected_file))
            
        print("✓ Scatter plots generated successfully")

    def test_generate_report(self):
        """Test report generation"""
        print("\nRunning test_generate_report...")
        
        # Generate report with different correlation types
        self.analyzer.generate_report(correlation_types=['kendall', 'spearman'])
        
        # Check if report file exists
        report_file = os.path.join(self.temp_dir, 'qpp_correlation_report.txt')
        self.assertTrue(os.path.exists(report_file))
        
        # Check report content
        with open(report_file, 'r') as f:
            content = f.read()
            self.assertIn("QPP Correlation Analysis Report", content)
            self.assertIn("KENDALL", content)
            self.assertIn("SPEARMAN", content)
            
        print("✓ Report generated successfully")

    def test_plot_correlations_boxplot(self):
        """Test correlation boxplot generation"""
        print("\nRunning test_plot_correlations_boxplot...")
        
        # Test with different correlation types
        for corr_type in ['kendall', 'spearman', 'pearson']:
            self.analyzer.plot_correlations_boxplot(correlation_type=corr_type, save_plot=True)
            expected_file = os.path.join(self.temp_dir, f'qpp_correlations_boxplot_{corr_type}.pdf')
            self.assertTrue(os.path.exists(expected_file))
            
        print("✓ Correlation boxplots generated successfully")

    def test_plot_correlations_across_datasets(self):
        """Test plotting correlations across multiple datasets"""
        print("\nRunning test_plot_correlations_across_datasets...")
        
        # Create a second analyzer with slightly different scores
        qpp_scores_2 = {
            qid: {method: score * 0.9 for method, score in scores.items()}
            for qid, scores in self.qpp_scores.items()
        }
        
        retrieval_metrics_2 = {
            metric: {
                "per_query": {qid: score * 0.9 for qid, score in scores["per_query"].items()},
                "mean": scores["mean"] * 0.9
            }
            for metric, scores in self.retrieval_metrics.items()
        }
        
        analyzer_2 = QPPCorrelationAnalyzer(
            qpp_scores_2,
            retrieval_metrics_2,
            output_dir=self.temp_dir
        )
        
        # Test plotting across datasets
        datasets = {
            "dataset1": self.analyzer,
            "dataset2": analyzer_2
        }
        
        QPPCorrelationAnalyzer.plot_correlations_across_datasets(
            datasets,
            correlation_type='kendall',
            output_dir=self.temp_dir
        )
        
        expected_file = os.path.join(self.temp_dir, 'qpp_correlations_across_datasets_kendall.pdf')
        self.assertTrue(os.path.exists(expected_file))
        
        print("✓ Cross-dataset correlation plot generated successfully")

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\nRunning test_edge_cases...")
        
        # Test with empty scores
        empty_scores = {}
        empty_metrics = {"ndcg@10": {"per_query": {}, "mean": 0.0}}
        
        with self.assertRaises(ValueError) as context:
            QPPCorrelationAnalyzer(empty_scores, empty_metrics)
        self.assertIn("QPP scores dictionary cannot be empty", str(context.exception))
        
        # Test with invalid metrics
        invalid_metrics = {"invalid_metric": {"per_query": {"q1": 0.5}, "mean": 0.5}}
        with self.assertRaises(ValueError) as context:
            QPPCorrelationAnalyzer(self.qpp_scores, invalid_metrics)
        self.assertIn("No valid metrics found", str(context.exception))
        
        # Test with mismatched QIDs
        mismatched_scores = {"q999": {"idf_avg": 0.5}}
        mismatched_metrics = {"ndcg@10": {"per_query": {"q1": 0.5}, "mean": 0.5}}
        
        analyzer = QPPCorrelationAnalyzer(mismatched_scores, mismatched_metrics)
        self.assertEqual(len(analyzer.qpp_df), 0)
        
        # Test with NaN values
        nan_scores = {"q1": {"idf_avg": np.nan}}
        nan_metrics = {"ndcg@10": {"per_query": {"q1": 0.5}, "mean": 0.5}}
        
        analyzer = QPPCorrelationAnalyzer(nan_scores, nan_metrics)
        correlations = analyzer.calculate_correlations(['kendall'])
        
        # Check if any valid correlation was computed
        self.assertTrue(
            all(not isinstance(v, (int, float)) or pd.isna(v) 
                for v in correlations['kendall'].values.flatten()),
            "Expected no valid correlations due to NaN values"
        )
        
        print("✓ Edge cases handled correctly")

    def test_align_qids(self):
        """Test QID alignment with IquiqueDataset"""
        print("\nRunning test_align_qids...")
        
        # Create test data based on IquiqueDataset
        qpp_scores = {
            "0": {"idf_avg": 0.8, "clarity": 0.7},  # Valid QID
            "1": {"idf_avg": 0.6, "clarity": 0.5},  # Valid QID
            "999": {"idf_avg": 0.4, "clarity": 0.3}, # Invalid QID
        }
        
        retrieval_metrics = {
            "ndcg@10": {
                "per_query": {
                    "0": 0.9,  # Valid QID
                    "2": 0.7,  # Valid QID but not in QPP scores
                    "3": 0.5   # Valid QID but not in QPP scores
                },
                "mean": 0.7
            },
            "ap": {
                "per_query": {
                    "0": 0.85,
                    "2": 0.65,
                    "3": 0.45
                },
                "mean": 0.65
            }
        }
        
        # Initialize analyzer with test data
        analyzer = QPPCorrelationAnalyzer(
            qpp_scores,
            retrieval_metrics,
            output_dir=self.temp_dir
        )
        
        # Check that only common QIDs are retained
        expected_qids = {"0"}  # Only QID "0" appears in both dictionaries
        self.assertEqual(set(analyzer.qpp_df.index), expected_qids)
        self.assertEqual(set(analyzer.metrics_df.index), expected_qids)
        
        # Check that invalid QIDs were removed
        self.assertNotIn("999", analyzer.qpp_df.index)
        self.assertNotIn("2", analyzer.qpp_df.index)
        self.assertNotIn("3", analyzer.qpp_df.index)
        
        # Check dimensions after alignment
        self.assertEqual(len(analyzer.qpp_df), len(expected_qids))
        self.assertEqual(len(analyzer.metrics_df), len(expected_qids))
        
        # Check that values are preserved for aligned QIDs
        self.assertEqual(analyzer.qpp_df.loc["0", "idf_avg"], 0.8)
        self.assertEqual(analyzer.metrics_df.loc["0", "ndcg@10"], 0.9)
        
        print("✓ QID alignment tested successfully")

    def test_align_qids_with_iquique_dataset(self):
        """Test QID alignment with actual IquiqueDataset"""
        print("\nRunning test_align_qids_with_iquique_dataset...")
        
        dataset = IquiqueDataset()
        
        # Create QPP scores for all queries in IquiqueDataset
        qpp_scores = {
            qid: {
                "idf_avg": 0.8 - float(qid) * 0.1,
                "clarity": 0.7 - float(qid) * 0.1
            }
            for qid in dataset.get_topics()['qid']
        }
        
        # Create retrieval metrics based on dataset qrels
        retrieval_metrics = {
            "ndcg@10": {
                "per_query": {
                    qid: 1.0 - float(qid) * 0.1
                    for qid in dataset.get_qrels()['qid'].unique()
                },
                "mean": 0.7
            },
            "ap": {
                "per_query": {
                    qid: 0.9 - float(qid) * 0.1
                    for qid in dataset.get_qrels()['qid'].unique()
                },
                "mean": 0.65
            }
        }
        
        # Initialize analyzer
        analyzer = QPPCorrelationAnalyzer(
            qpp_scores,
            retrieval_metrics,
            output_dir=self.temp_dir
        )
        
        # Check that all valid QIDs from dataset are present
        expected_qids = set(dataset.get_topics()['qid'])
        self.assertEqual(set(analyzer.qpp_df.index), expected_qids)
        self.assertEqual(set(analyzer.metrics_df.index), expected_qids)
        
        # Check dimensions
        self.assertEqual(len(analyzer.qpp_df), len(dataset.get_topics()))
        self.assertEqual(len(analyzer.metrics_df), len(dataset.get_topics()))
        
        # Check values for specific queries
        for qid in dataset.get_topics()['qid']:
            self.assertEqual(
                analyzer.qpp_df.loc[qid, "idf_avg"],
                0.8 - float(qid) * 0.1
            )
            self.assertEqual(
                analyzer.metrics_df.loc[qid, "ndcg@10"],
                1.0 - float(qid) * 0.1
            )
        
        print("✓ QID alignment with IquiqueDataset tested successfully")

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files after all tests"""
        import shutil
        shutil.rmtree(cls.temp_dir)

if __name__ == '__main__':
    unittest.main() 