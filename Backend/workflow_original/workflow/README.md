# 🔄 Workflow - Système de Vérification de Conformité ODDO

Version am

éliorée du système de vérification de conformité basée sur la revue de code.

## 📁 Structure du Projet

```
workflow/
├── adapters/
│   ├── __init__.py
│   └── rules_adapter.py          # Convertit JSON rules → format CAG
├── parsers/
│   ├── __init__.py
│   └── excel_parser.py            # Parse les fichiers Excel ODDO
├── config/
│   ├── __init__.py
│   └── settings.py                # Configuration centralisée
├── doc_analyzer.py                # Agent d'analyse documentaire (À CRÉER)
├── theorist.py                    # Agent d'orchestration (À CRÉER)
├── checker.py                     # Agent de vérification d'alignement (À CRÉER)
├── dis_glos.py                    # Agent disclaimers & glossaires (À CRÉER)
├── IMPLEMENTATION_PLAN.md         # Plan détaillé des modifications
└── README.md                      # Cette documentation
```

## 🚀 Démarrage Rapide

### 1. Parser les fichiers Excel (Étape préalable)

```bash
cd workflow
python parsers/excel_parser.py
```

**Sortie :**
- `glossaire_disclaimers_parsed.json`
- `registration_abroad_parsed.json`

### 2. Adapter les règles de conformité

```bash
python adapters/rules_adapter.py
```

**Sortie :**
- `rules_adapted_cag.json` (58 règles adaptées au format CAG)

### 3. Lancer les agents individuellement

```bash
# Agent doc_analyzer
python doc_analyzer.py

# Agent theorist
python theorist.py

# Agent checker
python checker.py

# Agent dis_glos  
python dis_glos.py
```

## 🔧 Modifications Principales

### ✅ Déjà Implémenté

| Composant | Description | Fichier |
|-----------|-------------|---------|
| **Parser Excel** | Parse GLOSSAIRE DISCLAIMERS & Registration abroad | `parsers/excel_parser.py` |
| **Adaptateur Règles** | Convertit `regles_contextuelles.json` + `regles_structurelles.json` → format CAG | `adapters/rules_adapter.py` |
| **Configuration** | Centralise tous les paramètres, chemins, clés API | `config/settings.py` |

### 🔄 En Cours

| Agent | Modifications Prévues |
|-------|----------------------|
| **doc_analyzer.py** | • Intégration `rules_adapter`<br>• Chargement règles JSON<br>• Configuration externalisée |
| **theorist.py** | • DefaultCache amélioré (TTL, max_size)<br>• Métriques réelles<br>• Batch processing conservé |
| **checker.py** | • BM25 + Hybrid search<br>• Seuils corriges (0.7)<br>• Chunking optimisé (600/200) |
| **dis_glos.py** | • Intégration Excel parser<br>• Détection langue améliorée<br>• Seuils augmentés (0.5) |

## ⚙️ Configuration

Tous les paramètres sont dans `config/settings.py` :

```python
# API
LLAMA_API_KEY = "sk-99be443a0c674b8297921465ab8e9510"
LLAMA_MODEL = "hosted_vllm/Llama-3.1-70B-Instruct"

# Seuils
SIMILARITY_THRESHOLD_STRICT = 0.7  # (modifié de 0.1)
DISCLAIMER_ALIGNMENT_THRESHOLD = 0.5  # (modifié de 0.3)
LLM_SCORE_THRESHOLD = 7  # (modifié de 5)

# Chunking
CHUNK_SIZE = 600  # (réduit de 800)
CHUNK_OVERLAP = 200  # (augmenté de 150)

# Batch
BATCH_SIZE = 5
PROCESS_ALL_ELEMENTS = True  # Traiter TOUS les éléments
```

## 📊 Améliorations Clés

### 1. Parser Excel Complet

- ✅ Parse **toutes les feuilles** automatiquement
- ✅ Détection intelligente des colonnes
- ✅ Export JSON structuré
- ✅ Gestion des erreurs robuste

### 2. Adaptateur de Règles Intelligent

- ✅ Mapping automatique `id` → `rule_id`
- ✅ Inférence `element_type` (table, image, paragraph, etc.)
- ✅ Détection sévérité basée sur mots-clés
- ✅ Préservation règle originale

### 3. Cache Amélioré (Theorist)

```python
class DefaultCache:
    - TTL (Time-To-Live): 1h
    - Max size: 1000 entrées
    - Métriques: hit/miss rate
    - LRU eviction
```

### 4. BM25 + Hybrid Search (Checker)

```python
# 3 scores calculés:
- semantic_score (embeddings)
- bm25_score (lexical)
- hybrid_score (0.5 * BM25 + 0.5 * semantic)
```

### 5. Intégration Excel (Dis_Glos)

- Disclaimers officiels chargés
- Pays autorisés par fonds
- Validation contre référence

## 🎯 Remarques Utilisateur Intégrées

1. ✅ **DefaultCache** : Implémenté avec TTL et metrics
2. ✅ **Règles JSON** : Adaptateur créé pour `regles_contextuelles.json` & `regles_structurelles.json`
3. ✅ **Batch Processing** : Conservé (évite hallucination LLM)
4. ✅ **Prompts Généraux** : Non ODDO-spécifiques, applicables partout
5. ✅ **Métriques** : Collecteur centralisé, export JSON
6. ✅ **TP/FP/FN** : Logique custom CONSERVÉE (force du checker)
7. ✅ **BM25** : Ajouté comme métrique supplémentaire (pas remplacement)
8. ✅ **Excel Parser** : Parse fichiers Excel entiers

## 📦 Dépendances Requises

```bash
pip install pandas openpyxl  # Pour Excel parser
pip install rank-bm25  # Pour BM25 dans checker
pip install langdetect  # Pour détection langue améliorée
```

## 🔍 Validation

Chaque agent peut être testé individuellement :

```bash
# Test avec document exemple
python doc_analyzer.py
# → Sortie: doc_analyzer_output.json

python theorist.py
# → Sortie: theorist_output.json + visualisations

python checker.py
# → Sortie: checker_output.json + métriques

python dis_glos.py
# → Sortie: dis_glos_output.json + rapports
```

## 📝 Notes Importantes

- Les clés API sont FIXÉES dans `config/settings.py` (pas d'arguments terminal)
- Tous les agents sont INDÉPENDANTS (peuvent être lancés séparément)
- Les chemins sont ABSOLUS dans settings.py (à adapter selon votre environnement)
- Les agents utilisent le même `llm_client.py` centralisé (à créer)

## 🏗️ Prochaines Étapes

1. Copier et modifier les 4 agents principaux
2. Créer `llm_client.py` centralisé (éviter duplication)
3. Tests individuels de chaque agent
4. Intégration finale dans orchest rateur unifié

---

**Version**: 1.0  
**Date**: 8 Décembre 2025  
**Basé sur**: Code Review v0.95
