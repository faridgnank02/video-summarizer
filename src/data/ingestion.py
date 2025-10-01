"""
Module d'ingestion de données vidéo/audio
Extraction de transcripts depuis YouTube et autres sources
"""

import os
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import requests
from pathlib import Path

import ssl
import urllib.request

# Désactiver la vérification SSL
ssl._create_default_https_context = ssl._create_unverified_context

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter
except ImportError:
    YouTubeTranscriptApi = None
    TextFormatter = None

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

logger = logging.getLogger(__name__)


@dataclass
class VideoData:
    """Structure pour stocker les données vidéo"""
    url: str
    title: str
    transcript: str
    duration: Optional[int] = None
    language: str = "en"
    source: str = "youtube"
    metadata: Optional[Dict[str, Any]] = None


class VideoIngestionError(Exception):
    """Exception personnalisée pour les erreurs d'ingestion"""
    pass


class YouTubeTranscriptExtractor:
    """Extracteur de transcripts YouTube"""
    
    def __init__(self, languages: List[str] = None):
        if YouTubeTranscriptApi is None:
            raise ImportError("youtube-transcript-api non installé. Installez avec: pip install youtube-transcript-api")
        
        self.languages = languages or ['fr', 'en', 'auto']
        self.formatter = TextFormatter()
    
    def extract_video_id(self, url: str) -> str:
        """Extrait l'ID vidéo depuis l'URL YouTube"""
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        else:
            raise VideoIngestionError(f"URL YouTube invalide: {url}")
    
    def get_transcript(self, video_url: str, language: str = None) -> VideoData:
        """
        Extrait le transcript d'une vidéo YouTube
        
        Args:
            video_url: URL de la vidéo YouTube
            language: Langue préférée (optionnel)
            
        Returns:
            VideoData: Données de la vidéo avec transcript
        """
        try:
            video_id = self.extract_video_id(video_url)
            
            # Définir les langues à essayer
            languages_to_try = [language] if language else self.languages
            
            transcript_list = None
            selected_language = None
            
            # Essayer d'obtenir le transcript dans les langues disponibles
            for lang in languages_to_try:
                try:
                    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=[lang])
                    selected_language = lang
                    break
                except Exception as e:
                    logger.debug(f"Transcript non disponible en {lang}: {e}")
                    continue
            
            # Si aucune langue spécifique ne fonctionne, essayer toutes les langues disponibles
            if transcript_list is None:
                try:
                    transcript_list = YouTubeTranscriptApi().fetch(video_id)
                    selected_language = transcript_list.language_code
                except Exception as e:
                    raise VideoIngestionError(f"Aucun transcript disponible pour la vidéo {video_id}: {e}")
            
            # Combiner les segments de transcript
            full_text = ""
            for snippet in transcript_list:
                full_text += snippet.text + " "
            
            # Obtenir les métadonnées de la vidéo (si possible)
            metadata = self._get_video_metadata(video_id)
            
            return VideoData(
                url=video_url,
                title=metadata.get('title', f'Vidéo {video_id}'),
                transcript=full_text.strip(),
                duration=metadata.get('duration'),
                language=selected_language,
                source='youtube',
                metadata=metadata
            )
            
        except Exception as e:
            raise VideoIngestionError(f"Erreur lors de l'extraction du transcript: {e}")
    
    def _get_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """Obtient les métadonnées de la vidéo (titre, durée, etc.)"""
        if yt_dlp is None:
            return {}
        
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extractaudio': False,
                'writesubtitles': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                
                return {
                    'title': info.get('title', ''),
                    'duration': info.get('duration'),
                    'view_count': info.get('view_count'),
                    'upload_date': info.get('upload_date'),
                    'uploader': info.get('uploader', ''),
                    'description': info.get('description', '')[:500]  # Limiter la description
                }
        except Exception as e:
            logger.warning(f"Impossible d'obtenir les métadonnées: {e}")
            return {}


class LocalVideoProcessor:
    """Processeur pour les fichiers vidéo locaux"""

    def __init__(self, model_path: str = "whisper_model", model_id: str = "large-v3"):
        try:
            import whisper
            self.whisper = whisper
            self.model_path = model_path
            self.model_id = model_id

            # Télécharger le modèle dans un chemin spécifique
            self.whisper_model = self.whisper.load_model(self.model_id, download_root=self.model_path)
        except ImportError:
            logger.warning("Whisper non installé. La transcription de fichiers locaux ne sera pas disponible.")
            self.whisper_model = None

    def extract_audio_transcript(self, video_path: str) -> VideoData:
        """Extrait le transcript d'un fichier vidéo local avec Whisper"""
        if self.whisper_model is None:
            raise VideoIngestionError("Whisper non installé. Installez avec: pip install openai-whisper")

        try:
            result = self.whisper_model.transcribe(video_path)

            return VideoData(
                url=video_path,
                title=Path(video_path).stem,
                transcript=result["text"],
                language=result.get("language", "unknown"),
                source="local_file"
            )
        except Exception as e:
            raise VideoIngestionError(f"Erreur lors de la transcription: {e}")


class DataIngestion:
    """Classe principale pour l'ingestion de données"""
    
    def __init__(self, languages: List[str] = None):
        self.youtube_extractor = YouTubeTranscriptExtractor(languages)
        self.local_processor = LocalVideoProcessor()
    
    def process_youtube_url(self, url: str, language: str = None) -> VideoData:
        """Traite une URL YouTube"""
        return self.youtube_extractor.get_transcript(url, language)
    
    def process_local_video(self, file_path: str) -> VideoData:
        """Traite un fichier vidéo local"""
        return self.local_processor.extract_audio_transcript(file_path)
    
    def process_text_input(self, text: str, title: str = "Texte personnalisé") -> VideoData:
        """Traite un texte fourni directement"""
        return VideoData(
            url="manual_input",
            title=title,
            transcript=text,
            source="manual"
        )
    
    def batch_process_urls(self, urls: List[str], language: str = None) -> List[VideoData]:
        """Traite plusieurs URLs en lot"""
        results = []
        for url in urls:
            try:
                data = self.process_youtube_url(url, language)
                results.append(data)
                logger.info(f"Transcript extrait avec succès pour: {data.title}")
            except VideoIngestionError as e:
                logger.error(f"Erreur pour {url}: {e}")
                continue
        
        return results


# Fonctions utilitaires pour la compatibilité avec le code existant
def get_transcript(video_url: str) -> str:
    """
    Fonction de compatibilité avec le code original du notebook
    """
    try:
        ingestion = DataIngestion()
        video_data = ingestion.process_youtube_url(video_url)
        return video_data.transcript
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    # Test rapide
    ingestion = DataIngestion()
    
    # Test avec une URL YouTube populaire
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll pour test
    try:
        data = ingestion.process_youtube_url(test_url)
        print(f"Titre: {data.title}")
        print(f"Transcript (100 premiers caractères): {data.transcript[:100]}...")
        print(f"Langue: {data.language}")
    except VideoIngestionError as e:
        print(f"Erreur: {e}")