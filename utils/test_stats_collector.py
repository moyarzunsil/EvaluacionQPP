import json
import os
import time
from datetime import datetime
from typing import Dict, Any
import unittest

class TestStatsCollector:
    """Utility class to collect and save test statistics"""
    
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = output_dir
        self.stats = {
            "fecha_ejecucion": "",
            "duracion_total": "",
            "total_pruebas": 0,
            "pruebas_exitosas": 0,
            "pruebas_fallidas": 0,
            "detalles_por_clase": []
        }
        os.makedirs(output_dir, exist_ok=True)
    
    def add_test_case(self, test_class: str, test_case: str, passed: bool, execution_time: float, details: Dict[str, Any] = None):
        # Buscar si la clase ya está en 'detalles_por_clase'
        class_stat = next((c for c in self.stats["detalles_por_clase"] if c["test_name"] == test_class), None)
        if not class_stat:
            class_stat = {
                "test_name": test_class,
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "execution_time": 0,
                "test_cases": []
            }
            self.stats["detalles_por_clase"].append(class_stat)
        
        # Agregar el caso de prueba
        class_stat["test_cases"].append({
            "name": test_case,
            "passed": passed,
            "execution_time": execution_time,
            "details": details or {}
        })
        
        # Update class stats
        class_stat["total_tests"] += 1
        if passed:
            class_stat["passed_tests"] += 1
            self.stats["pruebas_exitosas"] += 1
        else:
            class_stat["failed_tests"] += 1
            self.stats["pruebas_fallidas"] += 1
        
        class_stat["execution_time"] += execution_time
        
        # Update overall stats
        self.stats["total_pruebas"] += 1
    
    def set_overall_stats(self, fecha_ejecucion: str, duracion_total: str, total_pruebas: int = None, pruebas_exitosas: int = None, pruebas_fallidas: int = None):
        self.stats["fecha_ejecucion"] = fecha_ejecucion
        self.stats["duracion_total"] = duracion_total
        
        # Only update counts if explicitly provided
        if total_pruebas is not None:
            self.stats["total_pruebas"] = total_pruebas
        if pruebas_exitosas is not None:
            self.stats["pruebas_exitosas"] = pruebas_exitosas
        if pruebas_fallidas is not None:
            self.stats["pruebas_fallidas"] = pruebas_fallidas
    
    def save_stats(self):
        """Save test statistics to JSON file"""
        output_file = os.path.join(self.output_dir, "informe_detallado.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        return output_file

class StatsTestResult(unittest.TestResult):
    """Custom TestResult class that collects statistics"""
    
    def __init__(self, stats_collector, stream=None, descriptions=None, verbosity=None):
        super().__init__(stream, descriptions, verbosity)
        self.stats_collector = stats_collector
        self.successes = []
        self._mirrorOutput = False
        self.stream = stream
        self.descriptions = descriptions
        self.verbosity = verbosity

    def startTest(self, test):
        self._started_at = time.time()
        super().startTest(test)

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)
        execution_time = time.time() - self._started_at
        test_class = test.__class__.__name__
        test_method = test._testMethodName
        self.stats_collector.add_test_case(
            test_class,
            test_method,
            True,
            execution_time
        )

    def addError(self, test, err):
        super().addError(test, err)
        execution_time = time.time() - self._started_at
        test_class = test.__class__.__name__
        test_method = test._testMethodName
        self.stats_collector.add_test_case(
            test_class,
            test_method,
            False,
            execution_time,
            {"error": self._exc_info_to_string(err, test)}
        )

    def addFailure(self, test, err):
        super().addFailure(test, err)
        execution_time = time.time() - self._started_at
        test_class = test.__class__.__name__
        test_method = test._testMethodName
        self.stats_collector.add_test_case(
            test_class,
            test_method,
            False,
            execution_time,
            {"failure": self._exc_info_to_string(err, test)}
        )

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        execution_time = time.time() - self._started_at
        test_class = test.__class__.__name__
        test_method = test._testMethodName
        self.stats_collector.add_test_case(
            test_class,
            test_method,
            False,
            execution_time,
            {"skipped": reason}
        )