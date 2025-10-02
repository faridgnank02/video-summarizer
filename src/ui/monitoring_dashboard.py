"""
Dashboard de monitoring et administration pour Video Summarizer
Interface Streamlit avancée avec métriques temps réel
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ajouter le répertoire src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.metrics import MetricsCollector, AlertManager, setup_default_alerts
from evaluation.evaluator import SummaryEvaluator
from models.model_manager import ModelManager, ModelType

class MonitoringDashboard:
    """Dashboard de monitoring et administration"""
    
    def __init__(self):
        self.metrics_collector = None
        self.alert_manager = None
        self.evaluator = None
        self.model_manager = None
        
        self._init_components()
    
    def _init_components(self):
        """Initialise les composants de monitoring"""
        try:
            # Initialiser le collecteur de métriques
            if 'metrics_collector' not in st.session_state:
                st.session_state.metrics_collector = MetricsCollector()
                st.session_state.metrics_collector.start_collection()
            
            self.metrics_collector = st.session_state.metrics_collector
            
            # Gestionnaire d'alertes
            if 'alert_manager' not in st.session_state:
                st.session_state.alert_manager = AlertManager(self.metrics_collector)
                setup_default_alerts(st.session_state.alert_manager)
            
            self.alert_manager = st.session_state.alert_manager
            
            # Autres composants
            self.evaluator = SummaryEvaluator(load_models=False)  # Pas de modèles pour le dashboard
            self.model_manager = ModelManager()
            
        except Exception as e:
            st.error(f"Erreur initialisation dashboard: {e}")
    
    def render_dashboard(self):
        """Affiche le dashboard principal"""
        st.set_page_config(
            page_title="Video Summarizer - Monitoring",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📊 Video Summarizer - Dashboard de Monitoring")
        st.markdown("---")
        
        # Sidebar pour la navigation
        with st.sidebar:
            st.header("🎛️ Navigation")
            page = st.selectbox(
                "Choisir une page",
                ["🏠 Vue d'ensemble", "📈 Métriques système", "⚡ Performance", 
                 "📊 Évaluation", "🚨 Alertes", "🔧 Administration"]
            )
            
            # Contrôles globaux
            st.header("⚙️ Contrôles")
            auto_refresh = st.checkbox("Rafraîchissement auto", value=True)
            if auto_refresh:
                refresh_interval = st.slider("Intervalle (s)", 5, 60, 10)
                time.sleep(refresh_interval)
                st.rerun()
            
            if st.button("🔄 Actualiser maintenant"):
                st.rerun()
        
        # Rendu de la page sélectionnée
        if page == "🏠 Vue d'ensemble":
            self._render_overview()
        elif page == "📈 Métriques système":
            self._render_system_metrics()
        elif page == "⚡ Performance":
            self._render_performance_metrics()
        elif page == "📊 Évaluation":
            self._render_evaluation_dashboard()
        elif page == "🚨 Alertes":
            self._render_alerts_dashboard()
        elif page == "🔧 Administration":
            self._render_admin_dashboard()
    
    def _render_overview(self):
        """Vue d'ensemble du système"""
        st.header("🏠 Vue d'ensemble du système")
        
        # Métriques actuelles
        if self.metrics_collector:
            current_stats = self.metrics_collector.get_current_stats()
            
            # KPIs principaux
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🎯 Score de santé",
                    f"{current_stats.get('health', {}).get('score', 0)}/100",
                    delta=None
                )
            
            with col2:
                business = current_stats.get('business', {})
                st.metric(
                    "📊 Requêtes totales",
                    business.get('total_requests', 0),
                    delta=business.get('successful_requests', 0) - business.get('failed_requests', 0)
                )
            
            with col3:
                perf = current_stats.get('performance', {})
                st.metric(
                    "⚡ Taux de succès",
                    f"{perf.get('success_rate', 0):.1%}",
                    delta=None
                )
            
            with col4:
                system = current_stats.get('system', {})
                if system:
                    st.metric(
                        "💻 CPU",
                        f"{system.get('cpu_percent', 0):.1f}%",
                        delta=None
                    )
        
        # État des modèles
        st.subheader("🤖 État des modèles")
        
        if self.model_manager:
            col1, col2 = st.columns(2)
            
            with col1:
                led_available, led_msg = self.model_manager.is_model_available(ModelType.LED)
                st.success("✅ LED disponible") if led_available else st.error("❌ LED indisponible")
                st.caption(led_msg)
            
            with col2:
                openai_available, openai_msg = self.model_manager.is_model_available(ModelType.OPENAI)
                st.success("✅ OpenAI disponible") if openai_available else st.error("❌ OpenAI indisponible")
                st.caption(openai_msg)
        
        # Alertes actives
        st.subheader("🚨 Alertes actives")
        
        if self.alert_manager and self.alert_manager.active_alerts:
            for alert_name, alert_info in self.alert_manager.active_alerts.items():
                severity = alert_info['severity']
                timestamp = alert_info['timestamp']
                
                if severity == 'critical':
                    st.error(f"🔴 {alert_name} - {timestamp}")
                elif severity == 'warning':
                    st.warning(f"🟡 {alert_name} - {timestamp}")
                else:
                    st.info(f"🔵 {alert_name} - {timestamp}")
        else:
            st.success("✅ Aucune alerte active")
        
        # Activité récente
        st.subheader("📈 Activité récente")
        
        if self.metrics_collector:
            # Graphique d'activité des dernières heures
            historical_data = self.metrics_collector.get_historical_data(hours=6, metric_type="performance")
            
            if historical_data:
                df = pd.DataFrame(historical_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Graphique des temps de traitement
                fig = px.line(
                    df, 
                    x='timestamp', 
                    y='processing_time',
                    title="Temps de traitement (6 dernières heures)",
                    labels={'processing_time': 'Temps (s)', 'timestamp': 'Heure'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Pas de données d'activité récente")
    
    def _render_system_metrics(self):
        """Métriques système détaillées"""
        st.header("📈 Métriques système")
        
        if not self.metrics_collector:
            st.error("Collecteur de métriques non disponible")
            return
        
        # Contrôles temporels
        col1, col2 = st.columns(2)
        with col1:
            hours = st.selectbox("Période", [1, 6, 12, 24, 48], index=2)
        with col2:
            metric_types = st.multiselect(
                "Métriques à afficher",
                ["CPU", "Mémoire", "Disque", "Connexions"],
                default=["CPU", "Mémoire"]
            )
        
        # Récupérer les données
        historical_data = self.metrics_collector.get_historical_data(hours=hours, metric_type="system")
        
        if not historical_data:
            st.warning("Pas de données système disponibles")
            return
        
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Graphiques système
        if "CPU" in metric_types:
            fig_cpu = px.line(
                df, x='timestamp', y='cpu_percent',
                title=f"Utilisation CPU ({hours}h)",
                labels={'cpu_percent': 'CPU %', 'timestamp': 'Heure'}
            )
            fig_cpu.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Seuil critique")
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        if "Mémoire" in metric_types:
            fig_mem = px.line(
                df, x='timestamp', y='memory_percent',
                title=f"Utilisation mémoire ({hours}h)",
                labels={'memory_percent': 'Mémoire %', 'timestamp': 'Heure'}
            )
            fig_mem.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="Seuil critique")
            st.plotly_chart(fig_mem, use_container_width=True)
        
        if "Disque" in metric_types:
            fig_disk = px.line(
                df, x='timestamp', y='disk_usage_percent',
                title=f"Utilisation disque ({hours}h)",
                labels={'disk_usage_percent': 'Disque %', 'timestamp': 'Heure'}
            )
            st.plotly_chart(fig_disk, use_container_width=True)
        
        if "Connexions" in metric_types:
            fig_conn = px.line(
                df, x='timestamp', y='active_connections',
                title=f"Connexions actives ({hours}h)",
                labels={'active_connections': 'Connexions', 'timestamp': 'Heure'}
            )
            st.plotly_chart(fig_conn, use_container_width=True)
        
        # Statistiques récentes
        st.subheader("📊 Statistiques récentes")
        
        recent_data = df.tail(10)  # 10 dernières mesures
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CPU moyen", f"{recent_data['cpu_percent'].mean():.1f}%")
            st.metric("CPU max", f"{recent_data['cpu_percent'].max():.1f}%")
        
        with col2:
            st.metric("Mémoire moyenne", f"{recent_data['memory_percent'].mean():.1f}%")
            st.metric("Mémoire max", f"{recent_data['memory_percent'].max():.1f}%")
        
        with col3:
            st.metric("Disque", f"{recent_data['disk_usage_percent'].iloc[-1]:.1f}%")
            st.metric("Connexions", f"{recent_data['active_connections'].iloc[-1]}")
    
    def _render_performance_metrics(self):
        """Métriques de performance applicative"""
        st.header("⚡ Métriques de performance")
        
        if not self.metrics_collector:
            st.error("Collecteur de métriques non disponible")
            return
        
        # Contrôles
        hours = st.selectbox("Période d'analyse", [1, 6, 12, 24], index=1)
        
        # Données de performance
        perf_data = self.metrics_collector.get_historical_data(hours=hours, metric_type="performance")
        
        if not perf_data:
            st.warning("Pas de données de performance disponibles")
            return
        
        df = pd.DataFrame(perf_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Métriques globales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_ops = len(df)
            st.metric("🔢 Opérations totales", total_ops)
        
        with col2:
            success_rate = (df['success'].sum() / len(df)) * 100 if len(df) > 0 else 0
            st.metric("✅ Taux de succès", f"{success_rate:.1f}%")
        
        with col3:
            avg_time = df['processing_time'].mean() if len(df) > 0 else 0
            st.metric("⏱️ Temps moyen", f"{avg_time:.2f}s")
        
        with col4:
            models_used = df['model_name'].nunique() if len(df) > 0 else 0
            st.metric("🤖 Modèles utilisés", models_used)
        
        # Graphiques de performance
        st.subheader("📈 Analyse temporelle")
        
        # Temps de traitement dans le temps
        fig_time = px.scatter(
            df, x='timestamp', y='processing_time', color='model_name',
            title="Temps de traitement par opération",
            labels={'processing_time': 'Temps (s)', 'timestamp': 'Heure'}
        )
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Distribution des temps par modèle
        if len(df) > 0:
            fig_box = px.box(
                df, x='model_name', y='processing_time',
                title="Distribution des temps de traitement par modèle"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Analyse des erreurs
        st.subheader("🚨 Analyse des erreurs")
        
        errors_df = df[df['success'] == False]
        
        if len(errors_df) > 0:
            st.error(f"⚠️ {len(errors_df)} erreurs détectées")
            
            # Types d'erreurs
            error_counts = errors_df['error_message'].value_counts()
            
            fig_errors = px.bar(
                x=error_counts.values, y=error_counts.index,
                orientation='h',
                title="Types d'erreurs les plus fréquents"
            )
            st.plotly_chart(fig_errors, use_container_width=True)
            
            # Table des erreurs récentes
            st.subheader("Erreurs récentes")
            recent_errors = errors_df.sort_values('timestamp', ascending=False).head(10)
            st.dataframe(
                recent_errors[['timestamp', 'operation', 'model_name', 'error_message']],
                use_container_width=True
            )
        else:
            st.success("✅ Aucune erreur dans la période sélectionnée")
    
    def _render_evaluation_dashboard(self):
        """Dashboard d'évaluation de la qualité"""
        st.header("📊 Évaluation de la qualité")
        
        st.info("🚧 Fonctionnalité en développement - Dashboard d'évaluation avancé")
        
        # Interface pour évaluation manuelle
        st.subheader("🧪 Évaluation manuelle")
        
        with st.form("evaluation_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                original_text = st.text_area(
                    "Texte original",
                    height=200,
                    placeholder="Entrez le texte original ici..."
                )
            
            with col2:
                generated_summary = st.text_area(
                    "Résumé généré", 
                    height=200,
                    placeholder="Entrez le résumé à évaluer ici..."
                )
            
            reference_summary = st.text_area(
                "Résumé de référence (optionnel)",
                height=100,
                placeholder="Résumé de référence pour comparaison..."
            )
            
            model_name = st.selectbox(
                "Modèle utilisé",
                ["openai", "led", "autre"]
            )
            
            submitted = st.form_submit_button("🔍 Évaluer")
            
            if submitted and original_text and generated_summary:
                with st.spinner("Évaluation en cours..."):
                    try:
                        # Évaluation
                        report = self.evaluator.evaluate_summary(
                            original_text=original_text,
                            generated_summary=generated_summary,
                            reference_summary=reference_summary if reference_summary else None,
                            model_name=model_name
                        )
                        
                        # Affichage des résultats
                        st.success("✅ Évaluation terminée")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Score global", f"{report.metrics.overall_score:.3f}")
                        
                        with col2:
                            st.metric("Similarité sémantique", f"{report.metrics.semantic_similarity:.3f}")
                        
                        with col3:
                            st.metric("Cohérence", f"{report.metrics.coherence_score:.3f}")
                        
                        # Détails des métriques
                        st.subheader("📈 Métriques détaillées")
                        
                        metrics_data = {
                            "Métrique": ["ROUGE-1 F", "ROUGE-2 F", "ROUGE-L F", "Lisibilité", "Compression"],
                            "Score": [
                                report.metrics.rouge_1_f,
                                report.metrics.rouge_2_f,
                                report.metrics.rouge_l_f,
                                report.metrics.readability_score,
                                report.metrics.compression_ratio
                            ]
                        }
                        
                        df_metrics = pd.DataFrame(metrics_data)
                        
                        fig = px.bar(
                            df_metrics, x="Métrique", y="Score",
                            title="Distribution des métriques d'évaluation"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Recommandations
                        st.subheader("💡 Recommandations")
                        for rec in report.recommendations:
                            st.info(f"• {rec}")
                        
                    except Exception as e:
                        st.error(f"Erreur lors de l'évaluation: {e}")
    
    def _render_alerts_dashboard(self):
        """Dashboard des alertes"""
        st.header("🚨 Gestion des alertes")
        
        if not self.alert_manager:
            st.error("Gestionnaire d'alertes non disponible")
            return
        
        # Vérifier les alertes
        self.alert_manager.check_alerts()
        
        # Alertes actives
        st.subheader("⚠️ Alertes actives")
        
        if self.alert_manager.active_alerts:
            for alert_name, alert_info in self.alert_manager.active_alerts.items():
                severity = alert_info['severity']
                timestamp = alert_info['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                
                with st.expander(f"{severity.upper()}: {alert_name} - {timestamp}"):
                    st.json(alert_info['stats'])
        else:
            st.success("✅ Aucune alerte active")
        
        # Configuration des alertes
        st.subheader("⚙️ Configuration des alertes")
        
        with st.form("alert_config"):
            st.write("Seuils d'alerte")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cpu_threshold = st.slider("Seuil CPU (%)", 50, 95, 80)
                memory_threshold = st.slider("Seuil mémoire (%)", 60, 95, 85)
            
            with col2:
                error_rate_threshold = st.slider("Taux d'erreur max (%)", 5, 50, 20)
                response_time_threshold = st.slider("Temps de réponse max (s)", 30, 300, 120)
            
            if st.form_submit_button("💾 Sauvegarder configuration"):
                st.success("Configuration sauvegardée (simulation)")
    
    def _render_admin_dashboard(self):
        """Dashboard d'administration"""
        st.header("🔧 Administration du système")
        
        # Contrôles système
        st.subheader("🎛️ Contrôles système")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Redémarrer monitoring"):
                if self.metrics_collector:
                    self.metrics_collector.stop_collection()
                    time.sleep(1)
                    self.metrics_collector.start_collection()
                    st.success("Monitoring redémarré")
        
        with col2:
            if st.button("🧹 Nettoyer logs"):
                st.info("Nettoyage des logs (simulation)")
        
        with col3:
            if st.button("📊 Export métriques"):
                if self.metrics_collector:
                    data = self.metrics_collector.get_historical_data(hours=24)
                    df = pd.DataFrame(data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Télécharger CSV",
                        csv,
                        "metrics.csv",
                        "text/csv"
                    )
        
        # Configuration des modèles
        st.subheader("🤖 Configuration des modèles")
        
        if self.model_manager:
            # État des modèles
            models_status = {}
            for model_type in ModelType:
                available, message = self.model_manager.is_model_available(model_type)
                models_status[model_type.value] = {"available": available, "message": message}
            
            st.json(models_status)
        
        # Informations système
        st.subheader("ℹ️ Informations système")
        
        system_info = {
            "Version Python": sys.version,
            "Plateforme": os.name,
            "Répertoire de travail": os.getcwd(),
            "Variables d'environnement": {
                "OPENAI_API_KEY": "✅ Configurée" if os.getenv("OPENAI_API_KEY") else "❌ Manquante"
            }
        }
        
        st.json(system_info)

# Point d'entrée principal
def main():
    """Point d'entrée principal du dashboard"""
    dashboard = MonitoringDashboard()
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()