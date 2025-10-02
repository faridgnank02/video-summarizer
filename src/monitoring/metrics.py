"""
Système de monitoring avancé pour Video Summarizer
Collecte des métriques, logging, alertes et dashboards
"""

import os
import json
import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import sqlite3
from contextlib import contextmanager

import psutil

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Métriques système"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    gpu_memory_mb: Optional[float] = None
    active_connections: int = 0

@dataclass
class PerformanceMetrics:
    """Métriques de performance applicative"""
    timestamp: str
    operation: str
    model_name: str
    processing_time: float
    input_length: int
    output_length: int
    success: bool
    error_message: Optional[str] = None
    memory_peak_mb: float = 0.0

@dataclass
class BusinessMetrics:
    """Métriques business/utilisation"""
    timestamp: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_processing_time: float
    unique_users: int
    most_used_model: str
    total_characters_processed: int

class MetricsCollector:
    """Collecteur de métriques système et applicatives"""
    
    def __init__(self, 
                 db_path: str = "metrics.db",
                 collection_interval: int = 30,
                 retention_days: int = 30):
        """
        Initialise le collecteur de métriques
        
        Args:
            db_path: Chemin vers la base de données SQLite
            collection_interval: Intervalle de collecte en secondes
            retention_days: Durée de rétention des métriques
        """
        self.db_path = db_path
        self.collection_interval = collection_interval
        self.retention_days = retention_days
        self.running = False
        self.collector_thread = None
        
        # Buffers pour les métriques en temps réel
        self.performance_buffer = deque(maxlen=1000)
        self.system_buffer = deque(maxlen=1000)
        
        # Compteurs pour les métriques business
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.processing_times = deque(maxlen=100)
        self.users = set()
        self.model_usage = defaultdict(int)
        self.characters_processed = 0
        
        self._init_database()
    
    def _init_database(self):
        """Initialise la base de données SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_used_mb REAL,
                    disk_usage_percent REAL,
                    gpu_memory_mb REAL,
                    active_connections INTEGER
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    operation TEXT,
                    model_name TEXT,
                    processing_time REAL,
                    input_length INTEGER,
                    output_length INTEGER,
                    success BOOLEAN,
                    error_message TEXT,
                    memory_peak_mb REAL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS business_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_requests INTEGER,
                    successful_requests INTEGER,
                    failed_requests INTEGER,
                    avg_processing_time REAL,
                    unique_users INTEGER,
                    most_used_model TEXT,
                    total_characters_processed INTEGER
                )
            ''')
            
            # Index pour les performances
            conn.execute('CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_business_timestamp ON business_metrics(timestamp)')
    
    def start_collection(self):
        """Démarre la collecte automatique des métriques"""
        if self.running:
            return
        
        self.running = True
        self.collector_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.collector_thread.start()
        logger.info("Collecte de métriques démarrée")
    
    def stop_collection(self):
        """Arrête la collecte des métriques"""
        self.running = False
        if self.collector_thread:
            self.collector_thread.join(timeout=5)
        logger.info("Collecte de métriques arrêtée")
    
    def _collect_loop(self):
        """Boucle principale de collecte"""
        while self.running:
            try:
                # Collecter les métriques système
                system_metrics = self._collect_system_metrics()
                self.system_buffer.append(system_metrics)
                self._store_system_metrics(system_metrics)
                
                # Calculer et stocker les métriques business
                business_metrics = self._calculate_business_metrics()
                self._store_business_metrics(business_metrics)
                
                # Nettoyer les anciennes données
                self._cleanup_old_data()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Erreur lors de la collecte: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collecte les métriques système"""
        try:
            # Métriques CPU et mémoire
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU (si disponible)
            gpu_memory = None
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_memory = gpus[0].memoryUsed
            except (ImportError, Exception):
                # GPUtil non disponible ou aucun GPU détecté
                gpu_memory = None
            
            # Connexions réseau (avec gestion d'erreur pour macOS)
            active_connections = 0
            try:
                active_connections = len(psutil.net_connections())
            except (psutil.AccessDenied, OSError):
                # Permissions insuffisantes sur macOS/certains systèmes
                active_connections = 0
            
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                disk_usage_percent=disk.percent,
                gpu_memory_mb=gpu_memory,
                active_connections=active_connections
            )
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques système: {e}")
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                disk_usage_percent=0.0
            )
    
    def record_performance(self,
                          operation: str,
                          model_name: str,
                          processing_time: float,
                          input_length: int,
                          output_length: int,
                          success: bool,
                          error_message: Optional[str] = None,
                          user_id: Optional[str] = None):
        """Enregistre une métrique de performance"""
        
        # Métriques de performance
        metrics = PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            model_name=model_name,
            processing_time=processing_time,
            input_length=input_length,
            output_length=output_length,
            success=success,
            error_message=error_message,
            memory_peak_mb=psutil.Process().memory_info().rss / 1024 / 1024
        )
        
        self.performance_buffer.append(metrics)
        self._store_performance_metrics(metrics)
        
        # Mise à jour des compteurs business
        self.request_count += 1
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        self.processing_times.append(processing_time)
        self.model_usage[model_name] += 1
        self.characters_processed += input_length + output_length
        
        if user_id:
            self.users.add(user_id)
    
    def _calculate_business_metrics(self) -> BusinessMetrics:
        """Calcule les métriques business actuelles"""
        avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        most_used = max(self.model_usage.items(), key=lambda x: x[1])[0] if self.model_usage else "unknown"
        
        return BusinessMetrics(
            timestamp=datetime.now().isoformat(),
            total_requests=self.request_count,
            successful_requests=self.success_count,
            failed_requests=self.error_count,
            avg_processing_time=avg_time,
            unique_users=len(self.users),
            most_used_model=most_used,
            total_characters_processed=self.characters_processed
        )
    
    def _store_system_metrics(self, metrics: SystemMetrics):
        """Stocke les métriques système en base"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO system_metrics 
                (timestamp, cpu_percent, memory_percent, memory_used_mb, 
                 disk_usage_percent, gpu_memory_mb, active_connections)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp, metrics.cpu_percent, metrics.memory_percent,
                metrics.memory_used_mb, metrics.disk_usage_percent,
                metrics.gpu_memory_mb, metrics.active_connections
            ))
    
    def _store_performance_metrics(self, metrics: PerformanceMetrics):
        """Stocke les métriques de performance en base"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO performance_metrics 
                (timestamp, operation, model_name, processing_time, input_length,
                 output_length, success, error_message, memory_peak_mb)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp, metrics.operation, metrics.model_name,
                metrics.processing_time, metrics.input_length, metrics.output_length,
                metrics.success, metrics.error_message, metrics.memory_peak_mb
            ))
    
    def _store_business_metrics(self, metrics: BusinessMetrics):
        """Stocke les métriques business en base"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO business_metrics 
                (timestamp, total_requests, successful_requests, failed_requests,
                 avg_processing_time, unique_users, most_used_model, total_characters_processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp, metrics.total_requests, metrics.successful_requests,
                metrics.failed_requests, metrics.avg_processing_time, metrics.unique_users,
                metrics.most_used_model, metrics.total_characters_processed
            ))
    
    def _cleanup_old_data(self):
        """Nettoie les données anciennes selon la politique de rétention"""
        cutoff_date = (datetime.now() - timedelta(days=self.retention_days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM system_metrics WHERE timestamp < ?', (cutoff_date,))
            conn.execute('DELETE FROM performance_metrics WHERE timestamp < ?', (cutoff_date,))
            conn.execute('DELETE FROM business_metrics WHERE timestamp < ?', (cutoff_date,))
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques actuelles"""
        current_system = list(self.system_buffer)[-1] if self.system_buffer else None
        recent_performance = list(self.performance_buffer)[-10:] if self.performance_buffer else []
        
        # Statistiques des 10 dernières opérations
        if recent_performance:
            avg_time = sum(p.processing_time for p in recent_performance) / len(recent_performance)
            success_rate = sum(1 for p in recent_performance if p.success) / len(recent_performance)
        else:
            avg_time = 0
            success_rate = 0
        
        return {
            "system": asdict(current_system) if current_system else None,
            "performance": {
                "avg_processing_time": avg_time,
                "success_rate": success_rate,
                "recent_operations": len(recent_performance)
            },
            "business": asdict(self._calculate_business_metrics()),
            "health": self._calculate_health_score()
        }
    
    def _calculate_health_score(self) -> Dict[str, Any]:
        """Calcule un score de santé global du système"""
        current_system = list(self.system_buffer)[-1] if self.system_buffer else None
        
        if not current_system:
            return {"score": 0, "status": "unknown", "issues": ["Pas de données système"]}
        
        issues = []
        score = 100
        
        # Vérifications système
        if current_system.cpu_percent > 80:
            issues.append("CPU élevé")
            score -= 20
        
        if current_system.memory_percent > 85:
            issues.append("Mémoire élevée")
            score -= 20
        
        if current_system.disk_usage_percent > 90:
            issues.append("Disque plein")
            score -= 30
        
        # Vérifications performance
        recent_errors = sum(1 for p in list(self.performance_buffer)[-20:] if not p.success)
        if recent_errors > 5:
            issues.append("Taux d'erreur élevé")
            score -= 25
        
        # Déterminer le statut
        if score >= 80:
            status = "healthy"
        elif score >= 60:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "score": max(0, score),
            "status": status,
            "issues": issues
        }
    
    def get_historical_data(self, 
                           hours: int = 24,
                           metric_type: str = "system") -> List[Dict[str, Any]]:
        """Récupère les données historiques"""
        start_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        table_map = {
            "system": "system_metrics",
            "performance": "performance_metrics", 
            "business": "business_metrics"
        }
        
        table = table_map.get(metric_type, "system_metrics")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                f'SELECT * FROM {table} WHERE timestamp >= ? ORDER BY timestamp',
                (start_time,)
            )
            
            return [dict(row) for row in cursor.fetchall()]

class AlertManager:
    """Gestionnaire d'alertes pour le monitoring"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules = []
        self.active_alerts = {}
        self.notification_handlers = []
    
    def add_alert_rule(self, 
                      name: str,
                      condition: Callable[[Dict[str, Any]], bool],
                      severity: str = "warning",
                      cooldown_minutes: int = 15):
        """Ajoute une règle d'alerte"""
        self.alert_rules.append({
            "name": name,
            "condition": condition,
            "severity": severity,
            "cooldown_minutes": cooldown_minutes
        })
    
    def add_notification_handler(self, handler: Callable[[str, str, str], None]):
        """Ajoute un gestionnaire de notification"""
        self.notification_handlers.append(handler)
    
    def check_alerts(self):
        """Vérifie toutes les règles d'alerte"""
        current_stats = self.metrics_collector.get_current_stats()
        
        for rule in self.alert_rules:
            try:
                if rule["condition"](current_stats):
                    self._trigger_alert(rule, current_stats)
                else:
                    self._resolve_alert(rule["name"])
            except Exception as e:
                logger.error(f"Erreur vérification alerte {rule['name']}: {e}")
    
    def _trigger_alert(self, rule: Dict[str, Any], stats: Dict[str, Any]):
        """Déclenche une alerte"""
        alert_key = rule["name"]
        now = datetime.now()
        
        # Vérifier le cooldown
        if alert_key in self.active_alerts:
            last_alert = self.active_alerts[alert_key]["timestamp"]
            if (now - last_alert).total_seconds() < rule["cooldown_minutes"] * 60:
                return
        
        # Déclencher l'alerte
        self.active_alerts[alert_key] = {
            "timestamp": now,
            "severity": rule["severity"],
            "stats": stats
        }
        
        # Notifier
        message = f"Alerte {rule['severity']}: {rule['name']}"
        for handler in self.notification_handlers:
            try:
                handler(rule["name"], rule["severity"], message)
            except Exception as e:
                logger.error(f"Erreur notification: {e}")
    
    def _resolve_alert(self, alert_name: str):
        """Résout une alerte"""
        if alert_name in self.active_alerts:
            del self.active_alerts[alert_name]

# Gestionnaires de notification par défaut
def log_notification_handler(name: str, severity: str, message: str):
    """Handler de notification par log"""
    level = logging.ERROR if severity == "critical" else logging.WARNING
    logger.log(level, f"ALERT [{severity.upper()}] {name}: {message}")

def email_notification_handler(name: str, severity: str, message: str):
    """Handler de notification par email (à implémenter)"""
    # TODO: Implémenter l'envoi d'email
    logger.info(f"Email alert: {message}")

# Configuration par défaut des alertes
def setup_default_alerts(alert_manager: AlertManager):
    """Configure les alertes par défaut"""
    
    # Alerte CPU élevé
    alert_manager.add_alert_rule(
        name="CPU élevé",
        condition=lambda stats: stats.get("system", {}).get("cpu_percent", 0) > 80,
        severity="warning"
    )
    
    # Alerte mémoire élevée
    alert_manager.add_alert_rule(
        name="Mémoire élevée", 
        condition=lambda stats: stats.get("system", {}).get("memory_percent", 0) > 85,
        severity="critical"
    )
    
    # Alerte taux d'erreur élevé
    alert_manager.add_alert_rule(
        name="Taux d'erreur élevé",
        condition=lambda stats: stats.get("performance", {}).get("success_rate", 1) < 0.8,
        severity="critical"
    )
    
    # Alerte temps de traitement élevé
    alert_manager.add_alert_rule(
        name="Temps de traitement élevé",
        condition=lambda stats: stats.get("performance", {}).get("avg_processing_time", 0) > 120,
        severity="warning"
    )

if __name__ == "__main__":
    # Test du système de monitoring
    collector = MetricsCollector()
    collector.start_collection()
    
    # Simuler quelques opérations
    collector.record_performance(
        operation="summarize",
        model_name="openai",
        processing_time=2.5,
        input_length=1000,
        output_length=200,
        success=True,
        user_id="test_user"
    )
    
    time.sleep(2)
    
    # Afficher les stats
    stats = collector.get_current_stats()
    print(json.dumps(stats, indent=2))
    
    collector.stop_collection()