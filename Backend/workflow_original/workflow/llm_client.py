"""
Client LLM centralisé pour tous les agents du workflow
Évite la duplication de code et assure la cohérence
"""

import httpx
import json
import re
import time
from typing import Dict, List, Any, Optional
from openai import OpenAI

# Import configuration
import sys
sys.path.append('.')
from config.settings import *


class LlamaClient:
    """
    Client unifié pour Llama 3.1-70B-Instruct
    Utilisé par tous les agents du workflow
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialise le client Llama
        
        Args:
            api_key: Clé API (par défaut depuis settings.py)
            base_url: URL de base (par défaut depuis settings.py)
        """
        self.api_key = api_key or LLAMA_API_KEY
        self.base_url = base_url or LLAMA_BASE_URL
        self.model_name = LLAMA_MODEL
        
        # Client HTTP avec timeout
        self.http_client = httpx.Client(verify=False)
        
        # Client OpenAI
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=self.http_client
        )
        
        # Métriques
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0
        self.errors = 0
        
        print(f"✅ Client Llama initialisé: {self.model_name}")
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        retry_count: int = 2
    ) -> Optional[str]:
        """
        Génère une réponse textuelle
        
        Args:
            messages: Liste de messages [{role, content}]
            temperature: Température (défaut depuis settings)
            max_tokens: Tokens max (défaut depuis settings)
            retry_count: Nombre de tentatives
        
        Returns:
            Texte de la réponse ou None si échec
        """
        temperature = temperature if temperature is not None else LLM_TEMPERATURE
        max_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
        
        for attempt in range(retry_count):
            try:
                self.total_calls += 1
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=LLM_TOP_P,
                    frequency_penalty=LLM_FREQUENCY_PENALTY,
                    presence_penalty=LLM_PRESENCE_PENALTY
                )
                
                elapsed = time.time() - start_time
                self.total_time += elapsed
                
                content = response.choices[0].message.content
                
                # Tokens (si disponible)
                if hasattr(response, 'usage') and response.usage:
                    self.total_tokens += response.usage.total_tokens
                
                return content
            
            except Exception as e:
                self.errors += 1
                print(f"❌ Erreur LLM (tentative {attempt +1}/{retry_count}): {e}")
                
                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt
                    print(f"⏳ Nouvelle tentative dans {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Échec après {retry_count} tentatives")
                    return None
        
        return None
    
    def generate_json_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = None,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Génère une réponse JSON structurée
        
        Args:
            system_prompt: Prompt système
            user_prompt: Prompt utilisateur
            temperature: Température pour la génération
            max_retries: Nombre de tentatives
        
        Returns:
            Dict parsé ou dict avec erreur
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response_text = self.generate_response(messages, temperature=temperature, retry_count=max_retries)
        
        if response_text is None:
            return {"error": "Échec de l'appel LLM"}
        
        # Parser le JSON
        return self._parse_json_response(response_text)
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """
        Parse et nettoie une réponse JSON
        
        Args:
            content: Contenu textuel de la réponse
        
        Returns:
            Dict parsé
        """
        try:
            if not content:
                return {"error": "Réponse vide"}
            
            # Nettoyer la réponse
            cleaned = self._clean_json_response(content)
            
            # Extraire JSON
            json_data = self._extract_json_robust(cleaned)
            
            if json_data:
                return json_data
            else:
                return {"error": "Impossible de parser JSON", "raw_response": content[:500]}
        
        except Exception as e:
            return {"error": f"Erreur parsing: {e}", "raw_response": content[:500]}
    
    def _clean_json_response(self, content: str) -> str:
        """Nettoie la réponse pour extraction JSON"""
        if not content:
            return "{}"
        
        # Remplacements courants
        replacements = [
            ('```json', ''),
            ('```', ''),
            ('\\"', '"'),
            ('\\\\n', ' '),
            ('\\\\t', ' '),
        ]
        
        cleaned = content
        for old, new in replacements:
            cleaned = cleaned.replace(old, new)
        
        # Enlever caractères de contrôle
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
        
        return cleaned.strip()
    
    def _extract_json_robust(self, content: str) -> Optional[Dict[str, Any]]:
        """Extrait JSON avec plusieurs méthodes"""
        # Méthode 1: Trouver première paire {}
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = content[start:end]
                return json.loads(json_str)
        except:
            pass
        
        # Méthode 2: Regex
        try:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            pass
        
        # Méthode 3: Essayer directement
        try:
            return json.loads(content)
        except:
            pass
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques d'utilisation
        
        Returns:
            Dict avec statistiques d'utilisation
        """
        avg_time = self.total_time / self.total_calls if self.total_calls > 0 else 0
        
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_time_seconds": round(self.total_time, 2),
            "average_time_seconds": round(avg_time, 2),
            "errors": self.errors,
            "success_rate": round((self.total_calls - self.errors) / self.total_calls * 100, 2) if self.total_calls > 0 else 0
        }
    
    def reset_metrics(self):
        """Réinitialise les métriques"""
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0
        self.errors = 0
        print("🔄 Métriques réinitialisées")


# ====================  EXEMPLE D'UTILISATION ====================
if __name__ == "__main__":
    print("=" * 70)
    print("🤖 TEST CLIENT LLM CENTRALISÉ")
    print("=" * 70)
    
    # Initialisation
    client = LlamaClient()
    
    # Test 1: Réponse texte simple
    print("\n📝 Test 1: Génération de texte")
    print("-" * 70)
    
    messages = [
        {"role": "system", "content": "Tu es un assistant expert en analyse de documents."},
        {"role": "user", "content": "Explique en une phrase ce qu'est un disclaimer dans un document financier."}
    ]
    
    response = client.generate_response(messages, max_tokens=150)
    print(f"Réponse: {response}")
    
    # Test 2: Réponse JSON
    print("\n📋 Test 2: Génération JSON")
    print("-" * 70)
    
    system_prompt = "Tu es un expert en analyse. Réponds uniquement en JSON."
    user_prompt = """Analyse ce texte et retourne un JSON avec les clés: 
    'sentiment' (positif/neutre/négatif), 
    'mots_cles' (liste),
    'score_confiance' (0-1).
    
    Texte: "Ce produit financier offre une performance stable et des risques maîtrisés."
    """
    
    json_response = client.generate_json_response(system_prompt, user_prompt)
    print(f"JSON parsé: {json.dumps(json_response, indent=2, ensure_ascii=False)}")
    
    # Métriques
    print("\n📊 Métriques d'utilisation")
    print("-" * 70)
    metrics = client.get_metrics()
    for key, value in metrics.items():
        print(f"  • {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✅ TESTS TERMINÉS")
    print("=" * 70)
