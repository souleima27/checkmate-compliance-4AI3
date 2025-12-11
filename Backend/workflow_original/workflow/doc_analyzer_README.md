# ✅ doc_analyzer.py - Version Finale (Multi-Format)

## 📊 Inputs du Système (3 Inputs)

Le `doc_analyzer.py` accepte maintenant **3 inputs** comme requis :

### 1. 📄 Fichier Document (Multi-Format)
- **Formats Supportés**: 
  - ✅ **PPTX** (PowerPoint) : Texte, Tableaux, Images
  - ✅ **DOCX** (Word) : Texte, Tableaux, Images (via extraction ZIP)
  - ✅ **PDF** : Texte, Métadonnées (via PyPDF2)
- **Rôle**: Document à analyser pour conformité

### 2. 📋 Règles JSON (Contextuelles + Structurelles)
- **Fichiers**: `regles_contextuelles.json` + `regles_structurelles.json`
- **Chargement**: Automatique depuis `config/settings.py`

### 3. 🏷️ Métadonnées JSON
- **Fichier**: `metadata.json` (même dossier que document)
- **Rôle**: Détermine QUELLES règles sont applicables

## ✨ Nouvelles Fonctionnalités de Parsing

### 💾 Sauvegarde JSON Parsé
Le document est d'abord converti en structure JSON standardisée et **sauvegardé automatiquement**.
- **Chemin**: `workflow/output/parsed_docs/[nom_fichier]_parsed.json`
- **Utilité**: Permet de vérifier comment le document est "vu" par l'agent avant analyse.

### 📍 Identification Précise (Nouveau)
Chaque élément analysé possède maintenant un identifiant unique pour faciliter la localisation :
- **PPTX**: `Slide X - ID slide_X_shape_Y`
- **DOCX**: `ID para_X` ou `ID table_X`
- **PDF**: `Page X`

### 🔍 Parsing Spécialisé
Chaque format a son propre parser robuste :

1. **PPTX Parser**
   - Extrait slides, layouts
   - Détecte texte, tableaux, images
   - Structure hiérarchique (Slide > Shape > Content)

2. **DOCX Parser**
   - Extrait paragraphes, styles
   - Extrait tableaux
   - **Extraction Images**: Décompresse le DOCX pour trouver les images dans `word/media/`

3. **PDF Parser**
   - Extrait texte par page
   - Extrait métadonnées PDF

## 🔧 Utilisation

### Mode Standalone

```bash
cd workflow
python doc_analyzer.py
```

### Sortie Complète

L'agent génère maintenant 2 fichiers par analyse :

1. **Structure Parsée** (`output/parsed_docs/doc_parsed.json`)
```json
{
  "type": "pptx",
  "slides": [
    {
      "slide_number": 1,
      "content": [
        {"id": "slide_1_shape_1", "type": "text", "text": "Titre..."}
      ]
    }
  ]
}
```

2. **Résultat Analyse** (`output/doc_analyzer_output.json`)
```json
{
  "status": "success",
  "analysis": {
    "conformite_globale": { "score": 85, "niveau": "bon" },
    "elements_conformes": [
      {
        "element": "Titre",
        "location": "Slide 1 - slide_1_shape_1",
        "regle_id": "RC1",
        "justification": "..."
      }
    ],
    "elements_non_conformes": [
      {
        "element": "Disclaimer",
        "location": "Page 3",
        "regle_id": "RC2",
        "violation": "..."
      }
    ]
  }
}
```

## 📊 Métriques de Conformité (Mise à Jour)

Les métriques ont été redéfinies pour être plus pertinentes et applicables à tous les formats (PPTX, DOCX, PDF) :

### 1. Compliance Score (Score Global)
- **Formule**: `Conformes / Total Requis`
- **Signification**: Capacité globale du document à respecter les règles applicables.
- **Cible**: 100%

### 2. Completeness (Complétude)
- **Formule**: `(Conformes + Non-Conformes) / Total Requis`
- **Signification**: Est-ce que tous les éléments obligatoires sont présents ? (Même s'ils contiennent des erreurs).
- **Utilité**: Détecte les sections manquantes.

### 3. Correctness (Exactitude)
- **Formule**: `Conformes / (Conformes + Non-Conformes)`
- **Signification**: Parmi les éléments présents, quelle est la proportion sans erreur ?
- **Utilité**: Mesure la qualité du contenu existant.

### 4. LLM Score (Subjectif)
- **Source**: Évaluation directe par le modèle (0-100%)
- **Utilité**: Contre-vérification "humaine" simulée.

---

**Version**: 1.3 (Multi-Format + Location IDs)  
**Date**: 8 Décembre 2025  
**Status**: ✅ Complet et testé
