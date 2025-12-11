"""
THEORIST AGENT - Vérification Sémantique et Conformité Approfondie (LangGraph + Batching)

Cet agent analyse les documents parsés pour vérifier :
1. La cohérence sémantique (Graphe conceptuel simulé)
2. La qualité rédactionnelle et le lexique (Anglicismes)
3. La conformité stricte aux règles (Deep Dive)

Architecture:
- LangGraph pour l'orchestration
- Traitement par batchs (Slides/Pages) pour documents longs
- Cache dédié et métriques avancées
"""

import json
import os
import sys
import time
import hashlib
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
from langgraph.graph import StateGraph, END

# Import modules centralisés
sys.path.append('.')
from config.settings import *
from llm_client import LlamaClient
from workflow.doc_analyzer import ComplianceRuleManager

# ==================== CACHE DÉDIÉ ====================

class TheoristCache:
    """Gestionnaire de cache dédié pour l'agent Theorist"""
    
    def __init__(self):
        self.cache_dir = os.path.join(CACHE_DIR, "theorist_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get(self, doc_id: str, rules_hash: str) -> Optional[Dict]:
        """Récupère une analyse du cache"""
        cache_path = self._get_cache_path(doc_id, rules_hash)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Erreur lecture cache: {e}")
        return None
        
    def set(self, doc_id: str, rules_hash: str, data: Dict):
        """Sauvegarde une analyse dans le cache"""
        cache_path = self._get_cache_path(doc_id, rules_hash)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Erreur écriture cache: {e}")
            
    def _get_cache_path(self, doc_id: str, rules_hash: str) -> str:
        """Génère le chemin du fichier cache"""
        filename = f"{doc_id}_{rules_hash}.json"
        return os.path.join(self.cache_dir, filename)

# ==================== STATE LANGGRAPH ====================

class TheoristState(TypedDict):
    """État du graphe Theorist"""
    parsed_json_path: str
    doc_structure: Dict
    metadata: Dict
    rules_context: str
    doc_id: str
    rules_hash: str
    
    # Batching
    batches: List[str]
    batch_results: List[Dict]
    
    # Résultats
    final_analysis: Dict
    metrics: Dict
    start_time: float
    llm_duration: float

# ==================== AGENT THEORIST (LANGGRAPH) ====================

class TheoristAgent:
    """Agent de vérification sémantique et conformité avec LangGraph"""
    
    def __init__(self):
        print("\n" + "=" * 80)
        print("🧠 INITIALISATION THEORIST AGENT (LANGGRAPH + BATCHING)")
        print("=" * 80)
        
        self.llm_client = LlamaClient()
        self.cache = TheoristCache()
        self.rule_manager = ComplianceRuleManager()
        
        # Construction du Graphe
        self.workflow = self._build_workflow()
        
        print("✅ Theorist Agent prêt")

    def _build_workflow(self) -> StateGraph:
        """Construit le workflow LangGraph"""
        workflow = StateGraph(TheoristState)
        
        # Définition des nœuds
        workflow.add_node("load_data", self._load_data_node)
        workflow.add_node("check_cache", self._check_cache_node)
        workflow.add_node("prepare_batches", self._prepare_batches_node)
        workflow.add_node("analyze_batches", self._analyze_batches_node)
        workflow.add_node("aggregate_results", self._aggregate_results_node)
        workflow.add_node("calculate_metrics", self._calculate_metrics_node)
        
        # Définition des arêtes
        workflow.set_entry_point("load_data")
        
        workflow.add_edge("load_data", "check_cache")
        
        # Logique conditionnelle après check_cache
        def cache_condition(state):
            if state.get("final_analysis"):
                return "end" # Cache hit
            return "continue"
            
        workflow.add_conditional_edges(
            "check_cache",
            cache_condition,
            {
                "end": END,
                "continue": "prepare_batches"
            }
        )
        
        workflow.add_edge("prepare_batches", "analyze_batches")
        workflow.add_edge("analyze_batches", "aggregate_results")
        workflow.add_edge("aggregate_results", "calculate_metrics")
        workflow.add_edge("calculate_metrics", END)
        
        return workflow.compile()

    # ==================== NODES ====================

    def _load_data_node(self, state: TheoristState) -> TheoristState:
        """Charge le document et les métadonnées"""
        print("📥 Node: Chargement des données")
        start_time = time.time()
        
        # 1. Chargement Doc
        try:
            with open(state["parsed_json_path"], 'r', encoding='utf-8') as f:
                doc_structure = json.load(f)
        except Exception as e:
            print(f"❌ Erreur chargement JSON: {e}")
            return {"final_analysis": {"error": str(e)}}

        # 2. Métadonnées
        metadata = state.get("metadata") or doc_structure.get("metadata", {})
        
        # 3. Règles
        self.rule_manager.metadata = metadata
        self.rule_manager.applicable_rules = self.rule_manager._filter_applicable_rules(only_category='contextuelle')
        rules_context = self.rule_manager.format_rules_for_llm()
        
        # 4. Hash Cache
        contextual_rules = [r for r in self.rule_manager.all_rules.values() if r.get('categorie') == 'contextuelle']
        rules_str = json.dumps(contextual_rules, sort_keys=True)
        rules_hash = hashlib.md5(rules_str.encode('utf-8')).hexdigest()
        
        return {
            "doc_structure": doc_structure,
            "metadata": metadata,
            "rules_context": rules_context,
            "doc_id": doc_structure.get("document_id", "unknown"),
            "rules_hash": rules_hash,
            "start_time": start_time,
            "llm_duration": 0.0
        }

    def _check_cache_node(self, state: TheoristState) -> TheoristState:
        """Vérifie si l'analyse existe déjà en cache"""
        print("💾 Node: Vérification Cache")
        cached_result = self.cache.get(state["doc_id"], state["rules_hash"])
        
        if cached_result:
            print("  ✅ Résultat trouvé en cache")
            return {
                "final_analysis": cached_result["analysis"],
                "metrics": cached_result["metrics"]
            }
        return {}

    def _prepare_batches_node(self, state: TheoristState) -> TheoristState:
        """Découpe le document en batchs pour analyse"""
        print("✂️ Node: Préparation des batchs")
        doc_structure = state["doc_structure"]
        doc_type = doc_structure.get('type')
        batches = []
        
        # Stratégie de découpage selon format
        if doc_type == 'pptx':
            # Groupement par slides (ex: 5 slides par batch)
            slides = doc_structure.get('slides', [])
            batch_size = BATCH_SIZE # Depuis settings.py
            
            current_batch = []
            for slide in slides:
                slide_content = self._format_slide_content(slide)
                current_batch.append(slide_content)
                
                if len(current_batch) >= batch_size:
                    batches.append("\n".join(current_batch))
                    current_batch = []
            
            if current_batch:
                batches.append("\n".join(current_batch))
                
        elif doc_type == 'pdf':
            # Groupement par pages
            content = doc_structure.get('content', [])
            batch_size = 3 # Pages par batch
            
            current_batch = []
            for item in content:
                page_content = f"\n--- Page {item.get('page_number')} ---\n{item.get('text', '')}"
                current_batch.append(page_content)
                
                if len(current_batch) >= batch_size:
                    batches.append("\n".join(current_batch))
                    current_batch = []
            
            if current_batch:
                batches.append("\n".join(current_batch))
                
        else: # DOCX ou autre
            # Groupement par blocs de texte (ex: 4000 chars)
            content = doc_structure.get('content', [])
            current_batch = ""
            
            for item in content:
                item_text = f"ID: {item.get('id')} | {item.get('text', '')}\n"
                if len(current_batch) + len(item_text) > 4000:
                    batches.append(current_batch)
                    current_batch = item_text
                else:
                    current_batch += item_text
            
            if current_batch:
                batches.append(current_batch)
        
        print(f"  📊 {len(batches)} batchs créés")
        return {"batches": batches}

    def _analyze_batches_node(self, state: TheoristState) -> TheoristState:
        """Analyse chaque batch via LLM"""
        print("🧠 Node: Analyse des batchs (LLM)")
        batches = state["batches"]
        results = []
        total_llm_time = 0
        
        for i, batch_content in enumerate(batches):
            print(f"  Processing batch {i+1}/{len(batches)}...")
            
            prompt = THEORIST_ANALYSIS_PROMPT.format(
                doc_type=state["doc_structure"].get("type", "unknown"),
                metadata=json.dumps(state["metadata"], ensure_ascii=False, indent=2),
                rules_context=state["rules_context"],
                doc_content=batch_content
            )
            
            start_req = time.time()
            messages = [
                {"role": "system", "content": "Tu es un expert en conformité et linguistique."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_client.generate_response(
                messages=messages,
                temperature=0.1,
                max_tokens=LLM_MAX_TOKENS
            )
            total_llm_time += (time.time() - start_req)
            
            parsed = self._parse_llm_response(response)
            results.append(parsed)
            
        return {
            "batch_results": results,
            "llm_duration": total_llm_time
        }

    def _aggregate_results_node(self, state: TheoristState) -> TheoristState:
        """Agrège les résultats des batchs"""
        print("∑ Node: Agrégation des résultats")
        results = state["batch_results"]
        
        # Initialisation agrégat
        aggregated = {
            "semantic_analysis": {
                "consistency_score": 0,
                "drafting_score": 0,
                "lexical_score": 0,
                "anglicisms_detected": [],
                "inconsistencies": []
            },
            "compliance_details": [],
            "global_assessment": {
                "risk_level": "low",
                "summary": []
            }
        }
        
        count = len(results)
        if count == 0:
            return {"final_analysis": aggregated}
            
        # Somme des scores
        sem_cons = 0
        draft_qual = 0
        lex_comp = 0
        
        for res in results:
            sem = res.get("semantic_analysis", {})
            sem_cons += sem.get("consistency_score", 0)
            draft_qual += sem.get("drafting_score", 0)
            lex_comp += sem.get("lexical_score", 0)
            
            aggregated["semantic_analysis"]["anglicisms_detected"].extend(sem.get("anglicisms_detected", []))
            aggregated["semantic_analysis"]["inconsistencies"].extend(sem.get("inconsistencies", []))
            
            aggregated["compliance_details"].extend(res.get("compliance_details", []))
            
            summary = res.get("global_assessment", {}).get("summary", "")
            if summary:
                aggregated["global_assessment"]["summary"].append(summary)
        
        # Moyennes
        aggregated["semantic_analysis"]["consistency_score"] = round(sem_cons / count, 1)
        aggregated["semantic_analysis"]["drafting_score"] = round(draft_qual / count, 1)
        aggregated["semantic_analysis"]["lexical_score"] = round(lex_comp / count, 1)
        
        # Déduplication
        aggregated["semantic_analysis"]["anglicisms_detected"] = list(set(aggregated["semantic_analysis"]["anglicisms_detected"]))
        
        # Synthèse du résumé
        aggregated["global_assessment"]["summary"] = " | ".join(aggregated["global_assessment"]["summary"])[:1000]
        
        return {"final_analysis": aggregated}

    def _calculate_metrics_node(self, state: TheoristState) -> TheoristState:
        """Calcule les métriques finales et sauvegarde"""
        print("📊 Node: Calcul Métriques")
        analysis = state["final_analysis"]
        total_time = time.time() - state["start_time"]
        
        sem_analysis = analysis.get("semantic_analysis", {})
        
        # Métriques Performance
        perf_metrics = {
            "semantic_consistency": sem_analysis.get("consistency_score", 0),
            "drafting_quality": sem_analysis.get("drafting_score", 0),
            "lexical_compliance": sem_analysis.get("lexical_score", 0),
            "rule_confidence": 0.0
        }
        
        # Calcul confiance (FIXED TYPE ERROR)
        compliance_details = analysis.get("compliance_details", [])
        if compliance_details:
            # Conversion explicite en float pour éviter TypeError
            confidences = []
            for d in compliance_details:
                try:
                    conf = float(d.get("confidence", 0))
                    confidences.append(conf)
                except (ValueError, TypeError):
                    confidences.append(0.0)
            
            if confidences:
                perf_metrics["rule_confidence"] = round(sum(confidences) / len(confidences) * 100, 1)
            
        # Métriques Processus
        process_metrics = {
            "latency_seconds": round(total_time, 2),
            "llm_call_count": len(state.get("batches", [])),
            "llm_generation_time": round(state["llm_duration"], 2),
            "processing_overhead": round(total_time - state["llm_duration"], 2)
        }
        
        metrics = {"performance": perf_metrics, "process": process_metrics}
        
        # Sauvegarde Résultat
        result = {
            "status": "success",
            "document_id": state["doc_id"],
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis,
            "metrics": metrics["performance"],
            "process_metrics": metrics["process"]
        }
        
        self.cache.set(state["doc_id"], state["rules_hash"], result)
        
        # Sauvegarde Fichier
        input_name = os.path.basename(state["parsed_json_path"])
        output_filename = f"{os.path.splitext(input_name)[0]}_theorist.json"
        output_path = os.path.join(OUTPUT_DIR, "theorist_results", output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Analyse terminée: {output_path}")
        self._print_metrics_summary(metrics)
        
        return {"metrics": metrics}

    # ==================== HELPERS ====================

    def analyze_document(self, parsed_json_path: str, metadata_path: str = None, metadata: Dict = None) -> Dict[str, Any]:
        """Point d'entrée pour lancer le graphe"""
        initial_state = {
            "parsed_json_path": parsed_json_path,
            "metadata": metadata,
            # Autres champs initialisés par load_data
        }
        
        result = self.workflow.invoke(initial_state)
        return result.get("final_analysis", {})

    def _format_slide_content(self, slide: Dict) -> str:
        """Formate le contenu d'une slide"""
        slide_id = f"Slide {slide.get('slide_number')}"
        content = [f"\n--- {slide_id} ---"]
        
        for item in slide.get('content', []):
            item_id = item.get('id', 'N/A')
            text = item.get('text', '')
            if item.get('type') == 'table':
                text = f"[TABLEAU] {json.dumps(item.get('data', []))}"
            content.append(f"ID: {item_id} | {text}")
            
        return "\n".join(content)

    def _parse_llm_response(self, response: str) -> Dict:
        """Parse la réponse JSON du LLM"""
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return {"error": "No JSON found", "raw": response}
        except Exception as e:
            return {"error": str(e), "raw": response}

    def _print_metrics_summary(self, metrics: Dict):
        """Affiche un résumé des métriques"""
        perf = metrics["performance"]
        proc = metrics["process"]
        
        print("\n📊 MÉTRIQUES THEORIST:")
        print(f"  • Sémantique: {perf['semantic_consistency']}%")
        print(f"  • Rédaction:  {perf['drafting_quality']}%")
        print(f"  • Lexique:    {perf['lexical_compliance']}%")
        print(f"  • Latence:    {proc['latency_seconds']}s ({proc['llm_call_count']} appels)")

if __name__ == "__main__":
    agent = TheoristAgent()
    
    parsed_dir = os.path.join(OUTPUT_DIR, "parsed_docs")
    if os.path.exists(parsed_dir):
        files = [os.path.join(parsed_dir, f) for f in os.listdir(parsed_dir) if f.endswith("prospectus_parsed.json")]
        if files:
            latest_file = max(files, key=os.path.getctime)
            agent.analyze_document(latest_file)
        else:
            print("⚠️ Aucun fichier parsé trouvé.")
    else:
        print(f"⚠️ Dossier {parsed_dir} introuvable.")
