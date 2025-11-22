# 🎯 Guide Testeur - Video Summarizer

**Bienvenue dans Video Summarizer !** 👋

Cette application vous permet de générer automatiquement des résumés intelligents à partir de vidéos YouTube ou de texte direct, en utilisant des modèles d'IA locaux (Ollama) ou cloud (OpenAI).

---

## 🌐 Accès à l'Application

### **URL d'accès**
```
http://[IP_PUBLIQUE]:8501
```

**Remplacez `[IP_PUBLIQUE]` par l'adresse IP fournie par l'administrateur.**

### **Horaires de disponibilité**
- ⏰ **10h00 - 18h00 GMT** (tous les jours)
- 🌍 Convertir pour votre fuseau horaire :
  - Paris (CET) : 11h00 - 19h00
  - New York (EST) : 05h00 - 13h00
  - Tokyo (JST) : 19h00 - 03h00 (lendemain)

### **Premier accès**
⚠️ **Le premier démarrage peut prendre 2-3 minutes** (téléchargement du modèle Ollama). Si la page ne charge pas immédiatement, patientez quelques instants et rafraîchissez.

---

## 🎨 Interface de l'Application

L'application se compose de **3 onglets principaux** :

### 1️⃣ **YouTube** 🔗
Résumer des vidéos YouTube à partir de leur URL

### 2️⃣ **Local File** 📁
Upload de fichiers audio/vidéo (⚠️ fonctionnalité limitée dans cette version)

### 3️⃣ **Direct Text** 📝
Saisie directe de texte à résumer

---

## 📖 Guide d'Utilisation

### 🎬 **Option 1 : Résumer une vidéo YouTube**

#### **Étape 1 : Accéder à l'onglet YouTube**
- Cliquez sur l'onglet **"🔗 YouTube"**

#### **Étape 2 : Saisir l'URL**
- Copiez l'URL d'une vidéo YouTube
- Exemple : `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
- Collez-la dans le champ "YouTube URL"

#### **Étape 3 : Choisir la langue des sous-titres**
- Sélectionnez la langue préférée (Auto, English, French, etc.)
- **Important** : La vidéo doit avoir des sous-titres dans cette langue

#### **Étape 4 : Extraire la transcription**
- Cliquez sur **"📥 Extract Transcript"**
- Attendez quelques secondes
- La transcription s'affichera avec les informations de la vidéo

#### **Étape 5 : Configurer le résumé**
Dans la barre latérale (sidebar) à gauche :

**🤖 Summary Model :**
- **Auto (Recommended)** : Choix automatique du meilleur modèle
- **Ollama (Local - Free)** : Modèle local gemma3:1b, rapide et gratuit
- **OpenAI (Speed)** : GPT-4 via API, ultra-rapide et haute qualité

**📏 Summary Length :**
- **Long (200-500 words)** : Résumé détaillé en 5-8 phrases
- **Short (50-200 words)** : Résumé concis en 2-3 phrases

**🌍 Language :**
- **Auto-detect** : Détection automatique de la langue
- Ou sélectionnez manuellement (English, French, Spanish, German)

#### **Étape 6 : Générer le résumé**
- Cliquez sur **"🚀 Generate Summary"**
- Attendez la génération (3-15 secondes selon le modèle)
- Le résumé s'affiche avec des métriques (nombre de mots, compression, temps)

#### **Étape 7 : Exporter (optionnel)**
- **📄 Download TXT** : Télécharger en format texte
- **📊 Download JSON** : Télécharger en format JSON
- **📋 Copy** : Copier dans le presse-papiers

---

### 📝 **Option 2 : Résumer du texte direct**

#### **Étape 1 : Accéder à l'onglet Direct Text**
- Cliquez sur **"📝 Direct Text"**

#### **Étape 2 : Saisir votre texte**
- Collez ou tapez votre texte dans la zone de texte
- **Minimum** : 50 caractères
- **Maximum recommandé** : 10,000 caractères

#### **Étape 3 : Ajouter un titre (optionnel)**
- Donnez un titre à votre texte pour l'identifier facilement

#### **Étape 4 : Utiliser le texte**
- Cliquez sur **"📝 Use This Text"**

#### **Étape 5 : Configurer et générer**
- Suivez les mêmes étapes que pour YouTube (étapes 5-7)

---

## 🎯 Exemples de Tests

### **Test 1 : Vidéo YouTube en français**
```
URL : https://www.youtube.com/watch?v=exemple_fr
Langue sous-titres : French
Modèle : Ollama
Longueur : Long
Langue sortie : Auto-detect
```

### **Test 2 : Vidéo YouTube en anglais**
```
URL : https://www.youtube.com/watch?v=exemple_en
Langue sous-titres : English
Modèle : OpenAI
Longueur : Short
Langue sortie : English
```

### **Test 3 : Texte direct**
```
Texte : Copiez un article de presse (500-1000 mots)
Modèle : Auto
Longueur : Long
Langue sortie : Auto-detect
```

---

## ⚙️ Fonctionnalités Avancées

### **Barre latérale (Sidebar)**

#### **🧹 Memory Management**
Si l'application devient lente :
- **🗑️ Clear All** : Libère toute la mémoire des modèles
- **🔄 Unload Ollama** : Décharge uniquement le modèle Ollama

#### **📊 System Monitoring**
Affiche en temps réel :
- 💻 **CPU** : Utilisation processeur
- 🧠 **Memory** : Utilisation mémoire
- 💾 **Disk** : Utilisation disque

#### **ℹ️ Model Information**
Détails sur les modèles disponibles :
- Caractéristiques d'Ollama (local, gratuit, rapide)
- Caractéristiques d'OpenAI (cloud, payant, très rapide)

### **📚 Summary History**
- Tous vos résumés générés sont sauvegardés
- Accessible dans la section "📚 Summary History"
- Possibilité de **🗑️ Clear History** pour tout effacer

### **📈 Statistics**
Statistiques d'utilisation :
- Nombre total de requêtes
- Répartition par modèle (Ollama / OpenAI)
- Temps moyen de traitement

---

## ⚠️ Limitations et Problèmes Connus

### **Limitations fonctionnelles**

❌ **Fichiers locaux non supportés**
- La transcription audio Whisper n'est pas activée dans cette version
- Utilisez YouTube ou texte direct uniquement

❌ **Taille maximale du texte**
- Texte : ~10,000 caractères maximum
- Au-delà, le texte sera tronqué automatiquement

❌ **Sous-titres YouTube requis**
- La vidéo DOIT avoir des sous-titres dans la langue choisie
- Les vidéos sans sous-titres ne fonctionneront pas

### **Problèmes courants**

#### 🐛 **"Ollama (Unavailable)"**
**Cause** : Le serveur Ollama n'a pas démarré correctement

**Solution** :
1. Attendez 2-3 minutes (premier démarrage)
2. Rafraîchissez la page
3. Si le problème persiste, utilisez "Auto" ou "OpenAI"

#### 🐛 **"Error during extraction"**
**Cause** : Sous-titres non disponibles ou URL invalide

**Solution** :
1. Vérifiez que l'URL est correcte
2. Changez la langue des sous-titres
3. Essayez "Auto" pour la langue

#### 🐛 **"API error" ou "OpenAI (Unavailable)"**
**Cause** : Clé API OpenAI non configurée ou invalide

**Solution** :
- Utilisez "Ollama" ou "Auto" à la place
- Contactez l'administrateur si OpenAI est nécessaire

#### 🐛 **Application lente ou freeze**
**Cause** : Mémoire saturée

**Solution** :
1. Allez dans **Memory Management** (sidebar)
2. Cliquez sur **"🗑️ Clear All"**
3. Attendez quelques secondes
4. Réessayez votre requête

#### 🐛 **"Connection timeout"**
**Cause** : Service arrêté (hors horaires 10h-18h GMT)

**Solution** :
- Vérifiez l'heure GMT
- Réessayez pendant les horaires de disponibilité
- Contactez l'administrateur si urgent

---

## 📊 Comparaison des Modèles

| Caractéristique | **Ollama (gemma3:1b)** | **OpenAI (GPT-4)** |
|-----------------|------------------------|---------------------|
| **Vitesse** | ⚡ 3-10s | ⚡⚡ 2-5s |
| **Qualité** | 🌟🌟🌟🌟 | 🌟🌟🌟🌟🌟 |
| **Coût** | 🆓 Gratuit | 💰 Payant |
| **Internet** | ❌ Non requis | ✅ Requis |
| **Langues** | 🌍 Multi-langue | 🌍 Multi-langue |
| **Textes longs** | ✅ Excellent | ✅ Excellent |
| **Vie privée** | 🔒 100% local | ☁️ Cloud |

### **Quand utiliser Ollama ?**
- ✅ Tests gratuits et illimités
- ✅ Données sensibles (traitement local)
- ✅ Pas de connexion internet fiable

### **Quand utiliser OpenAI ?**
- ✅ Qualité maximale requise
- ✅ Vitesse critique
- ✅ Résumés très courts et précis

---

## 🧪 Scénarios de Test Recommandés

### **Test Basique (5 min)**
1. ✅ Résumer une vidéo YouTube courte (<5 min)
2. ✅ Résumer un texte direct (300 mots)
3. ✅ Tester les deux longueurs (Short + Long)
4. ✅ Exporter en TXT et JSON

### **Test Avancé (15 min)**
1. ✅ Tester les 2 modèles (Ollama + OpenAI)
2. ✅ Vidéos en français et anglais
3. ✅ Textes longs (>2000 mots)
4. ✅ Vérifier l'historique
5. ✅ Tester Clear All / Unload

### **Test de Performance (10 min)**
1. ✅ 5 résumés consécutifs
2. ✅ Mesurer les temps de génération
3. ✅ Vérifier le monitoring (CPU/RAM)
4. ✅ Tester après Clear All

### **Test de Robustesse (10 min)**
1. ✅ URL YouTube invalide
2. ✅ Vidéo sans sous-titres
3. ✅ Texte trop court (<50 caractères)
4. ✅ Texte très long (>10,000 caractères)
5. ✅ Changements rapides de modèle

---

## 📝 Formulaire de Feedback

### **Ce que nous voulons savoir :**

#### **1. Expérience Générale (1-5 étoiles)**
- Interface intuitive ?
- Facilité d'utilisation ?
- Satisfaction globale ?

#### **2. Qualité des Résumés**
- **Ollama** : Qualité, pertinence, cohérence ?
- **OpenAI** : Qualité, pertinence, cohérence ?
- Préférence entre Short et Long ?

#### **3. Performance**
- Temps de génération acceptable ?
- Application réactive ?
- Bugs ou freezes rencontrés ?

#### **4. Fonctionnalités**
- Fonctionnalités les plus utiles ?
- Fonctionnalités manquantes ?
- Améliorations suggérées ?

#### **5. Bugs Rencontrés**
- Description du problème
- Étapes pour reproduire
- Screenshots (si possible)

#### **6. Questions Ouvertes**
- Cas d'usage envisagés ?
- Améliorations prioritaires ?
- Commentaires libres

---

## 📧 Support et Contact

### **Problèmes Techniques**
- Consultez d'abord la section **"Problèmes Courants"** ci-dessus
- Si non résolu, contactez : [EMAIL_ADMIN]

### **Suggestions et Feedback**
- Formulaire de feedback : [LIEN_FORMULAIRE]
- Email : [EMAIL_FEEDBACK]

### **Informations Supplémentaires**
- Documentation technique : `docs/TECHNICAL_DOCUMENTATION.md`
- Dépôt GitHub : https://github.com/faridgnank02/video-summarizer
- Branch actuelle : `ollama-integration`

---

## 🎉 Merci de Tester !

Votre feedback est **essentiel** pour améliorer Video Summarizer. N'hésitez pas à :

✅ Tester toutes les fonctionnalités  
✅ Pousser les limites du système  
✅ Signaler le moindre bug  
✅ Proposer des améliorations  

**Bon test !** 🚀

---

## 📚 Annexes

### **Exemples d'URLs YouTube pour Tests**

**Vidéos courtes (2-5 min) :**
- TED Talks courts
- Tutoriels YouTube
- Actualités

**Vidéos moyennes (10-15 min) :**
- Conférences
- Documentaires courts
- Cours en ligne

**Vidéos longues (30+ min) :**
- Podcasts
- Conférences complètes
- Documentaires

### **Exemples de Textes pour Tests**

**Texte court (200-500 mots) :**
```
[Copiez un article de presse court]
```

**Texte moyen (500-1500 mots) :**
```
[Copiez un article de blog ou une page Wikipedia]
```

**Texte long (2000+ mots) :**
```
[Copiez un article académique ou un long rapport]
```

---

**Version du guide** : 1.0.0  
**Date de création** : 21 novembre 2025  
**Application** : Video Summarizer (branch: ollama-integration)
