"""
Configuration centralisée pour le système de vérification de conformité
"""

import os

# ====================  CHEMINS DES FICHIERS ====================
# Utilisation de chemins relatifs pour la portabilité

# Base dir: inside Backend/workflow
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR) # Backend/

# Dossiers de sortie
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
CACHE_DIR = os.path.join(BASE_DIR, "caches")

# Règles de conformité
REGLES_CONTEXTUELLES_PATH = os.path.join(CACHE_DIR, "regles_contextuelles.json")
REGLES_STRUCTURELLES_PATH = os.path.join(CACHE_DIR, "regles_structurelles.json")

# Fichiers de données (JSON)
GLOSSAIRE_DISCLAIMERS_PATH = os.path.join(CACHE_DIR, "disclaimers.json")
FOND_REGISTRED_PATH_JSON = os.path.join(CACHE_DIR, "fond_registred.json")
GLOSSAIRES_JSON_PATH = os.path.join(CACHE_DIR, "glossaires.json")

# Documents de test
TEST_DOCUMENT_PATH = os.path.join(BASE_DIR, "tests")



# ====================  API CONFIGURATION ====================

# IMPORTANT: En production, utiliser des variables d'environnement
# Pour l'instant, valeurs fixées dans le code comme demandé

LLAMA_API_KEY = "YOUR_API_KEY"
LLAMA_BASE_URL = "https://tokenfactory.esprit.tn/api"
LLAMA_MODEL = "hosted_vllm/Llama-3.1-70B-Instruct"


# ====================  PARAMÈTRES LLM ====================

LLM_TEMPERATURE = 0.1  # Température basse pour plus de déterminisme
LLM_MAX_TOKENS = 4000
LLM_TOP_P = 0.9
LLM_FREQUENCY_PENALTY = 0.1
LLM_PRESENCE_PENALTY = 0.1


# ====================  PARAMÈTRES OCR ====================

OCR_LANGUAGES = "fra+eng"  # Tesseract langues
OCR_CONFIG = r'--oem 3 --psm 6'
OCR_LANGUAGES = "fra+eng"  # Tesseract langues
OCR_CONFIG = r'--oem 3 --psm 6'
OCR_MIN_CONFIDENCE = 0.5  # Confiance minimale pour accepter le texte
ENABLE_OCR = False  # Toggle pour activer/désactiver l'OCR (True = traite images, False = ignore)


# ====================  PARAMÈTRES CHUNKING ====================

CHUNK_SIZE = 600  # Taille des chunks (réduit de 800 à 600)
CHUNK_OVERLAP = 200  # Overlap entre chunks (augmenté de 150 à 200)


# ====================  SEUILS DE SIMILARITÉ ====================

# Pour checker.py - Seuils CORRIGÉS (plus stricts)
SIMILARITY_THRESHOLD_STRICT = 0.7  # Threshold strict pour validation
SIMILARITY_THRESHOLD_FLEXIBLE = 0.4  # Threshold flexible pour recherche

# Pour dis_glos.py - Scores composites
DISCLAIMER_ALIGNMENT_THRESHOLD = 0.5  # Augmenté de 0.3 à 0.5
LLM_SCORE_THRESHOLD = 7  # Score LLM minimum (sur 10) - augmenté de 5 à 7


# ====================  PARAMÈTRES BATCH PROCESSING ====================

# Pour theorist.py
BATCH_SIZE = 5  # Nombre d'éléments par batch
PROCESS_ALL_ELEMENTS = True  # Ne PAS limiter le nombre de batchs (traiter TOUS les éléments)


# ====================  PARAMÈTRES MÉTRIQUES ====================

# Métriques de validation
MIN_ANSWER_LENGTH = 20  # Longueur minimale d'une réponse valide (augmenté de 5 à 20)


# ====================  PARAMÈTRES DE CACHE ====================

CACHE_ENABLED = True
CACHE_TTL = 3600  # Time-to-live en secondes (1 heure)
CACHE_MAX_SIZE = 1000  # Nombre maximum d'entrées


# ====================  LOGGING ====================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_FILE_BACKUP_COUNT = 5


# ====================  PROMPTS GÉNÉRAUX (NON ODDO-SPÉCIFIQUES) ====================

SYSTEM_PROMPT_DOCUMENT_ANALYSIS = """Tu es un expert en analyse de conformité documentaire générale.
Tu analyses des structures de documents et identifies les éléments à vérifier selon des règles de conformité.
Tu retournes UNIQUEMENT du JSON valide sans commentaires supplémentaires.

Tes analyses doivent être:
1. Basées uniquement sur les règles fournies
2. Précises et vérifiables
3. Applicables à tout type de document financier
4. Structurées et cohérentes"""

SYSTEM_PROMPT_ALIGNMENT_CHECK = """Tu es un assistant expert en vérification d'alignement de documents.
Tu compares deux versions d'un même document et identifies les différences importantes.

Tes réponses doivent:
1. Identifier les incohérences de contenu (hors langue et date)
2. Signaler les différences de données chiffrées
3. Être factuelles et basées sur les textes fournis
4. Suivre le format demandé strictement"""

SYSTEM_PROMPT_DISCLAIMER_GLOSSARY = """Tu es un assistant expert en analyse de disclaimers et glossaires.
Tu vérifies la conformité des mentions légales et la cohérence des termes techniques.

Format de réponse attendu:
- Pour les vérifications: [STATUT] - [RAISON]
- Pour les extractions: Liste claire et structurée
- Pour les analyses: Score suivi d'une justification brève"""


# ====================  CRÉATION AUTOMATIQUE DES DOSSIERS ====================

def create_directories():
    """Crée les dossiers nécessaires s'ils n'existent pas"""
    dirs = [OUTPUT_DIR, LOGS_DIR]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    print(f"✅ Dossiers créés: {dirs}")


#if __name__ == "__main__":
#    print("=" * 70)
#    print("⚙️  CONFIGURATION DU SYSTÈME")
#    print("=" * 70)
#    
#    print(f"\n📁 Chemins configurés:")
#    print(f"  • Règles contextuelles: {REGLES_CONTEXTUELLES_PATH}")
#    print(f"  • Règles structurelles: {REGLES_STRUCTURELLES_PATH}")
#    print(f"  • Glossaire: {GLOSSAIRE_DISCLAIMERS_PATH}")
#    print(f"  • Registration abroad: {REGISTRATION_ABROAD_PATH}")
#    
#    print(f"\n🤖 Configuration LLM:")
#    print(f"  • Modèle: {LLAMA_MODEL}")
#    print(f"  • Température: {LLM_TEMPERATURE}")
#    print(f"  • Max tokens: {LLM_MAX_TOKENS}")
#    
#    print(f"\n📊 Paramètres:")
#    print(f"  • Chunk size: {CHUNK_SIZE}")
#    print(f"  • Chunk overlap: {CHUNK_OVERLAP}")
#    print(f"  • Similarity threshold: {SIMILARITY_THRESHOLD_STRICT}")
#    print(f"  • Batch size: {BATCH_SIZE}")
#    
#    print(f"\n📂 Création des dossiers...")
#    create_directories()
#    
#    print("\n" + "=" * 70)
#    print("✅ CONFIGURATION CHARGÉE")
#    print("=" * 70)


# ==================== PROMPTS THEORIST ====================

THEORIST_ANALYSIS_PROMPT = """
ANALYSE SÉMANTIQUE ET CONFORMITÉ APPROFONDIE

Tu es un expert en conformité financière et analyse sémantique.
Ton objectif est de vérifier la cohérence, la qualité rédactionnelle et le respect strict des règles.

=== CONTEXTE DOCUMENTAIRE ===
Type: {doc_type}
Métadonnées: {metadata}

=== RÈGLES À VÉRIFIER ===
{rules_context}

=== CONTENU DU DOCUMENT (Extrait structuré) ===
{doc_content}

=== TÂCHE : GRAPHE SÉMANTIQUE & CONFORMITÉ ===
Analyse ce contenu en simulant un graphe conceptuel pour détecter les incohérences.

1. **Analyse Sémantique & Rédaction** :
   - Qualité du langage (Ton professionnel, clarté).
   - **Lexique** : Détecte TOUS les anglicismes non traduits (ex: "Track record" au lieu de "Historique").
   - **Cohérence** : Est-ce que les informations se contredisent ? (ex: "Risque faible" page 1 vs "Volatilité élevée" page 5).

2. **Vérification des Règles (Deep Dive)** :
   - Pour chaque règle fournie, vérifie si elle est respectée.
   - Cite PRECISEMENT l'ID de l'élément (ex: slide_1_shape_2) pour chaque preuve.

=== FORMAT DE RÉPONSE JSON ===
Réponds UNIQUEMENT en JSON :
{{
  "semantic_analysis": {{
    "consistency_score": 85, // 0-100
    "drafting_score": 90, // 0-100
    "lexical_score": 95, // 0-100
    "anglicisms_detected": ["track record", "benchmark"],
    "inconsistencies": [
      {{"description": "Contradiction risque", "location_1": "slide_1...", "location_2": "slide_5..."}}
    ]
  }},
  "compliance_details": [
    {{
      "rule_id": "RC1",
      "status": "compliant|non_compliant",
      "evidence": "Le disclaimer est présent...",
      "location": "slide_1_shape_5",
      "confidence": 0.95
    }}
  ],
  "global_assessment": {{
    "risk_level": "low|medium|high",
    "summary": "Document cohérent mais quelques anglicismes..."
  }}
}}
"""
CHECKER_VERIFY_ALIGNMENT_PROMPT = """
Tu es un auditeur de conformité. Tu dois vérifier si une présentation (Target) est alignée avec un document de référence, en répondant à une question spécifique.

QUESTION: {question}

RÉPONSE ATTENDUE (Selon Référence): {expected_answer}

CONTENU PERTINENT DE LA PRÉSENTATION (Target):
{context}

TÂCHE:
1. Cherche la réponse à la question dans le contenu pertinent de la présentation.
2. Compare cette réponse avec la réponse attendue.
3. Détermine le niveau d'alignement.

Génère 3 versions de réponse :
- Conservatrice : Strictement basée sur le texte fourni.
- Équilibrée : Synthétique et contextuelle.
- Créative : Déduit les implications (sans inventer).

Ensuite, choisis la MEILLEURE réponse qui représente fidèlement le contenu de la présentation.

Format de sortie JSON attendu:
{{
  "actual_answer": "La présentation mentionne que...",
  "justification": "Trouvé dans la slide X...",
  "alignment_status": "aligned|partial|misaligned|missing",
  "confidence": 0.9
}}
"""
