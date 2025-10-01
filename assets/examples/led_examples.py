#!/usr/bin/env python3
"""
Exemples d'utilisation du modèle LED pour le résumé de texte
"""

import sys
from pathlib import Path

# Ajouter src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.led_model import LEDSummarizer

def example_simple_summary():
    """Exemple de résumé simple"""
    print("📝 Exemple: Résumé simple")
    
    led = LEDSummarizer()
    
    text = """
    L'intelligence artificielle (IA) transforme rapidement notre monde moderne.
    Cette technologie permet aux ordinateurs de réaliser des tâches qui nécessitaient
    auparavant l'intelligence humaine. Les applications incluent la reconnaissance
    vocale, la traduction automatique, la conduite autonome et l'analyse de données.
    Cependant, l'IA soulève des questions importantes sur l'emploi, l'éthique et
    la protection de la vie privée que nous devons aborder de manière responsable.
    """
    
    summary = led.summarize(text, summary_type="short")
    print(f"Résumé: {summary}")

def example_long_document():
    """Exemple avec un document plus long"""
    print("\n📖 Exemple: Document long")
    
    led = LEDSummarizer()
    
    # Simuler un document plus long
    long_text = """
    L'intelligence artificielle représente l'une des avancées technologiques les plus
    significatives de notre époque. Cette discipline, qui combine informatique,
    mathématiques et sciences cognitives, vise à créer des systèmes capables de
    réaliser des tâches nécessitant normalement l'intelligence humaine.
    
    Les applications de l'IA sont désormais omniprésentes dans notre quotidien.
    Les assistants vocaux comme Siri, Alexa ou Google Assistant utilisent des
    algorithmes de reconnaissance vocale et de traitement du langage naturel.
    Les plateformes de streaming comme Netflix ou Spotify emploient des systèmes
    de recommandation basés sur l'apprentissage automatique.
    
    Dans le domaine médical, l'IA aide au diagnostic précoce de maladies,
    à l'analyse d'images médicales et au développement de nouveaux médicaments.
    L'industrie automobile développe des véhicules autonomes grâce à la vision
    par ordinateur et aux réseaux de neurones profonds.
    
    Cependant, cette révolution technologique soulève des défis importants.
    L'impact sur l'emploi inquiète de nombreux secteurs, car l'automatisation
    pourrait remplacer certains métiers. Les questions de protection des données
    personnelles et de vie privée deviennent cruciales avec la collecte massive
    d'informations par les algorithmes d'IA.
    
    L'avenir de l'intelligence artificielle dépend de notre capacité à développer
    cette technologie de manière éthique et responsable, en tenant compte des
    implications sociales, économiques et philosophiques qu'elle représente.
    """
    
    summary = led.summarize(long_text, summary_type="long")
    print(f"Résumé long: {summary}")

if __name__ == "__main__":
    example_simple_summary()
    example_long_document()
