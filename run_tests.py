import unittest
import os
import json
from datetime import datetime
from functools import partial
from utils.test_stats_collector import StatsTestResult, TestStatsCollector
import sys

def generate_markdown_report(stats, output_file):
    """Generate a markdown report from test statistics"""
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("# Unit Test Report\n\n")
        
        # Summary section
        f.write("## Summary\n\n")
        f.write(f"- **Execution Date:** {stats['fecha_ejecucion']}\n")
        f.write(f"- **Total Duration:** {stats['duracion_total']}\n")
        f.write(f"- **Total Tests:** {stats['total_pruebas']}\n")
        f.write(f"- **Passed Tests:** {stats['pruebas_exitosas']}\n")
        f.write(f"- **Failed Tests:** {stats['pruebas_fallidas']}\n\n")
        
        # Progress bar
        if stats['total_pruebas'] > 0:
            success_rate = (stats['pruebas_exitosas'] / stats['total_pruebas']) * 100
            progress_bar = "█" * int(success_rate / 2) + "░" * (50 - int(success_rate / 2))
            f.write(f"```\n{progress_bar} {success_rate:.1f}%\n```\n\n")
        
        # Detailed results section
        f.write("## Details by Test Class\n\n")
        
        for test_class in stats['detalles_por_clase']:
            # Class header
            f.write(f"### {test_class['test_name']}\n\n")
            
            # Class statistics
            f.write("#### Statistics\n\n")
            f.write(f"- **Tests Run:** {test_class['total_tests']}\n")
            f.write(f"- **Passed Tests:** {test_class['passed_tests']}\n")
            f.write(f"- **Failed Tests:** {test_class['failed_tests']}\n")
            f.write(f"- **Execution Time:** {test_class['execution_time']:.2f} seconds\n\n")
            
            # Test cases
            if test_class['test_cases']:
                f.write("#### Test Cases\n\n")
                f.write("| Status | Test | Time (s) |\n")
                f.write("|:------:|--------|------------|\n")
                
                for case in test_class['test_cases']:
                    status = "✅" if case['passed'] else "❌"
                    f.write(f"| {status} | `{case['name']}` | {case['execution_time']:.3f} |\n")
                    
                    if not case['passed'] and case['details']:
                        f.write(f"\n<details><summary>Error details</summary>\n\n")
                        f.write("```\n")
                        f.write(str(case['details']))
                        f.write("\n```\n</details>\n\n")
            
            f.write("\n---\n\n")

def run_all_tests():
    """Run all tests and generate a comprehensive report"""
    # Create the test results directory if it does not exist
    test_results_dir = "test_results"
    os.makedirs(test_results_dir, exist_ok=True)
    
    # Create a stats collector for the complete test execution
    overall_stats = TestStatsCollector(output_dir=test_results_dir)
    
    # Create the test runner with the custom result class using functools.partial
    runner = unittest.TextTestRunner(
        verbosity=2,
        resultclass=partial(StatsTestResult, overall_stats),
        stream=sys.stdout
    )
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    suite = loader.discover('EvaluacionQPP/tests')
    
    # Record start and end times
    start_time = datetime.now()
    result = runner.run(suite)
    end_time = datetime.now()
    
    # Set the overall statistics
    overall_stats.set_overall_stats(
        fecha_ejecucion=start_time.strftime("%Y-%m-%d %H:%M:%S"),
        duracion_total=str(end_time - start_time)
    )
    
    # Save detailed statistics to JSON
    overall_stats.save_stats()
    
    # Generate a readable report in English
    summary_file = os.path.join(test_results_dir, "test_report.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Test Report\n")
        f.write("===========\n\n")
        f.write(f"Execution Date: {overall_stats.stats['fecha_ejecucion']}\n")
        f.write(f"Total Duration: {overall_stats.stats['duracion_total']}\n\n")
        f.write("Summary:\n")
        f.write(f"- Total tests run: {overall_stats.stats['total_pruebas']}\n")
        f.write(f"- Successful tests: {overall_stats.stats['pruebas_exitosas']}\n")
        f.write(f"- Failed tests: {overall_stats.stats['pruebas_fallidas']}\n\n")
        
        f.write("Details by Test Class:\n")
        f.write("----------------------\n")
        for test_class in overall_stats.stats['detalles_por_clase']:
            f.write(f"\nClass: {test_class['test_name']}\n")
            f.write(f"- Tests run: {test_class['total_tests']}\n")
            f.write(f"- Successful tests: {test_class['passed_tests']}\n")
            f.write(f"- Failed tests: {test_class['failed_tests']}\n")
            f.write(f"- Execution time: {test_class['execution_time']:.2f} seconds\n")
            
            if test_class['test_cases']:
                f.write("\nTest cases:\n")
                for case in test_class['test_cases']:
                    status = "✓" if case['passed'] else "✗"
                    f.write(f"{status} {case['name']} ({case['execution_time']:.3f}s)\n")
                    if not case['passed'] and case['details']:
                        f.write(f"  Error details: {case['details']}\n")
            f.write("\n" + "-"*50 + "\n")
    
    # Generate markdown report
    markdown_file = os.path.join(test_results_dir, "test_report.md")
    generate_markdown_report(overall_stats.stats, markdown_file)
    
    print(f"\nReports generated at:")
    print(f"- TXT: {summary_file}")
    print(f"- Markdown: {markdown_file}")
    print(f"- JSON: {os.path.join(test_results_dir, 'informe_detallado.json')}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)