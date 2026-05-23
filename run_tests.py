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
        f.write("# Informe de Pruebas Unitarias\n\n")
        
        # Summary section
        f.write("## Resumen\n\n")
        f.write(f"- **Fecha de Ejecución:** {stats['fecha_ejecucion']}\n")
        f.write(f"- **Duración Total:** {stats['duracion_total']}\n")
        f.write(f"- **Total de Pruebas:** {stats['total_pruebas']}\n")
        f.write(f"- **Pruebas Exitosas:** {stats['pruebas_exitosas']}\n")
        f.write(f"- **Pruebas Fallidas:** {stats['pruebas_fallidas']}\n\n")
        
        # Progress bar
        if stats['total_pruebas'] > 0:
            success_rate = (stats['pruebas_exitosas'] / stats['total_pruebas']) * 100
            progress_bar = "█" * int(success_rate / 2) + "░" * (50 - int(success_rate / 2))
            f.write(f"```\n{progress_bar} {success_rate:.1f}%\n```\n\n")
        
        # Detailed results section
        f.write("## Detalles por Clase de Prueba\n\n")
        
        for test_class in stats['detalles_por_clase']:
            # Class header
            f.write(f"### {test_class['test_name']}\n\n")
            
            # Class statistics
            f.write("#### Estadísticas\n\n")
            f.write(f"- **Pruebas Ejecutadas:** {test_class['total_tests']}\n")
            f.write(f"- **Pruebas Exitosas:** {test_class['passed_tests']}\n")
            f.write(f"- **Pruebas Fallidas:** {test_class['failed_tests']}\n")
            f.write(f"- **Tiempo de Ejecución:** {test_class['execution_time']:.2f} segundos\n\n")
            
            # Test cases
            if test_class['test_cases']:
                f.write("#### Casos de Prueba\n\n")
                f.write("| Estado | Prueba | Tiempo (s) |\n")
                f.write("|:------:|--------|------------|\n")
                
                for case in test_class['test_cases']:
                    status = "✅" if case['passed'] else "❌"
                    f.write(f"| {status} | `{case['name']}` | {case['execution_time']:.3f} |\n")
                    
                    if not case['passed'] and case['details']:
                        f.write(f"\n<details><summary>Detalles del error</summary>\n\n")
                        f.write("```\n")
                        f.write(str(case['details']))
                        f.write("\n```\n</details>\n\n")
            
            f.write("\n---\n\n")

def run_all_tests():
    """Run all tests and generate a comprehensive report"""
    # Crear el directorio de resultados de pruebas si no existe
    test_results_dir = "test_results"
    os.makedirs(test_results_dir, exist_ok=True)
    
    # Crear un recolector de estadísticas para la ejecución completa de pruebas
    overall_stats = TestStatsCollector(output_dir=test_results_dir)
    
    # Crear el ejecutor de pruebas con la clase personalizada de resultados usando functools.partial
    runner = unittest.TextTestRunner(
        verbosity=2,
        resultclass=partial(StatsTestResult, overall_stats),
        stream=sys.stdout
    )
    
    # Descubrir y ejecutar todas las pruebas
    loader = unittest.TestLoader()
    suite = loader.discover('EvaluacionQPP/tests')
    
    # Registrar tiempos de inicio y fin
    start_time = datetime.now()
    result = runner.run(suite)
    end_time = datetime.now()
    
    # Establecer las estadísticas generales
    overall_stats.set_overall_stats(
        fecha_ejecucion=start_time.strftime("%Y-%m-%d %H:%M:%S"),
        duracion_total=str(end_time - start_time)
    )
    
    # Guardar las estadísticas detalladas en JSON
    overall_stats.save_stats()
    
    # Generar un informe legible en español
    summary_file = os.path.join(test_results_dir, "informe_de_pruebas.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Informe de Pruebas\n")
        f.write("==================\n\n")
        f.write(f"Fecha de Ejecución: {overall_stats.stats['fecha_ejecucion']}\n")
        f.write(f"Duración Total: {overall_stats.stats['duracion_total']}\n\n")
        f.write("Resumen:\n")
        f.write(f"- Total de pruebas ejecutadas: {overall_stats.stats['total_pruebas']}\n")
        f.write(f"- Pruebas exitosas: {overall_stats.stats['pruebas_exitosas']}\n")
        f.write(f"- Pruebas fallidas: {overall_stats.stats['pruebas_fallidas']}\n\n")
        
        f.write("Detalles por Clase de Prueba:\n")
        f.write("-----------------------------\n")
        for test_class in overall_stats.stats['detalles_por_clase']:
            f.write(f"\nClase: {test_class['test_name']}\n")
            f.write(f"- Pruebas ejecutadas: {test_class['total_tests']}\n")
            f.write(f"- Pruebas exitosas: {test_class['passed_tests']}\n")
            f.write(f"- Pruebas fallidas: {test_class['failed_tests']}\n")
            f.write(f"- Tiempo de ejecución: {test_class['execution_time']:.2f} segundos\n")
            
            if test_class['test_cases']:
                f.write("\nCasos de prueba:\n")
                for case in test_class['test_cases']:
                    status = "✓" if case['passed'] else "✗"
                    f.write(f"{status} {case['name']} ({case['execution_time']:.3f}s)\n")
                    if not case['passed'] and case['details']:
                        f.write(f"  Detalles del error: {case['details']}\n")
            f.write("\n" + "-"*50 + "\n")
    
    # Generate markdown report
    markdown_file = os.path.join(test_results_dir, "informe_de_pruebas.md")
    generate_markdown_report(overall_stats.stats, markdown_file)
    
    print(f"\nInformes generados en:")
    print(f"- TXT: {summary_file}")
    print(f"- Markdown: {markdown_file}")
    print(f"- JSON: {os.path.join(test_results_dir, 'informe_detallado.json')}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)