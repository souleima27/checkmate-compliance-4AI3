import os
# Force CPU usage to avoid CUDA OOM errors on small GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import shutil
import sys
import json
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from workflow.llm_client import LlamaClient

# Ensure 'utils' is importable
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from utils.conversion import convert_to_pdf
from utils.injection import inject_violations

llm = LlamaClient()

class InjectionModel(BaseModel):
    fileName: str
    violations: List[dict]


def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# Ensure 'workflow' directory is in PYTHONPATH so internal imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "workflow"))

# Import both agents
from workflow.doc_analyzer import DocumentComplianceAgent
from workflow.theorist import TheoristAgent
from workflow.new_dis_glos import OptimizedDisclaimerGlossaryAgent
from workflow.checker_memory import DocumentAgentWithMemory
from workflow.feedback_manager import FeedbackManager

feedback_manager = FeedbackManager()

class FeedbackModel(BaseModel):
    type: str # "chatbot" | "violation"
    id: str
    feedback: str # "like" | "dislike"
    details: Optional[dict] = {}


app = FastAPI(title="Compliance Check API")

@app.post("/api/feedback")
async def submit_feedback(data: FeedbackModel):
    """Submit user feedback."""
    try:
        entry = feedback_manager.add_feedback(data.dict())
        return {"status": "success", "entry": entry}
    except Exception as e:
        raise HTTPException(500, f"Error saving feedback: {str(e)}")




# Configure CORS (development mode)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "workflow", "output")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "workflow", "caches")
METADATA_FILE = os.path.join(CACHE_DIR, "metadata.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve uploaded files statically
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Initialize agents
print("⏳ Initializing AI Agents (loading models)...")

doc_analyzer_agent = None
theorist_agent = None
dis_glos_agent = None
checker_agent = None  # Global variable for the checker agent

try:
    doc_analyzer_agent = DocumentComplianceAgent()
    print("✅ DocumentComplianceAgent initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize DocumentComplianceAgent: {e}")

try:
    theorist_agent = TheoristAgent()
    print("✅ TheoristAgent initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize TheoristAgent: {e}")

try:
    dis_glos_agent = OptimizedDisclaimerGlossaryAgent()
    print("✅ OptimizedDisclaimerGlossaryAgent initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize OptimizedDisclaimerGlossaryAgent: {e}")


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Compliance Backend is running (LibreOffice + Injection Enabled)",
    }

# --- NEW ENDPOINTS ---

@app.get("/api/convert/{filename}")
async def convert_file(filename: str):
    """Convert PPTX/DOCX to PDF for preview using LibreOffice."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")
        
    output_dir = os.path.join(CACHE_DIR, "conversions")
    try:
        # This uses local LibreOffice or falls back (if implemented)
        pdf_path = convert_to_pdf(file_path, output_dir)
        return FileResponse(pdf_path, media_type="application/pdf")
    except FileNotFoundError as e:
        print(f"Conversion failed (File/Tool not found): {e}")
        raise HTTPException(424, detail=f"Conversion tool not found: {str(e)}")
    except Exception as e:
        print(f"Conversion failed: {e}")
        raise HTTPException(500, detail=f"Conversion failed: {str(e)}")

@app.post("/api/download-annotated")
async def download_annotated(data: InjectionModel):
    """
    Inject violations into the original file and return it.
    Expects fileName and violations list in body.
    """
    file_path = os.path.join(UPLOAD_DIR, data.fileName)
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")
        
    output_dir = os.path.join(OUTPUT_DIR, "annotated")
    try:
        output_path = inject_violations(file_path, data.violations, output_dir)
        filename = os.path.basename(output_path)
        # Return as attachment
        return FileResponse(output_path, filename=filename, media_type="application/octet-stream")
    except Exception as e:
        print(f"Injection failed: {e}")
        raise HTTPException(500, f"Injection failed: {str(e)}")

# ---------------------

@app.post("/api/save_metadata")
async def save_metadata(request: Request):
    """Save metadata into workflow/caches/metadata.json"""
    try:
        data = await request.json()
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"💾 Metadata saved to {METADATA_FILE}")
        return {"status": "success", "path": METADATA_FILE}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save metadata: {e}")


@app.post("/api/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
):
    """
    Receive a document and trigger the compliance analysis workflow.
    Runs BOTH doc_analyzer (structural) and theorist (contextual) agents.
    """

    if not doc_analyzer_agent:
        raise HTTPException(status_code=503, detail="DocumentComplianceAgent not initialized.")

    try:
        # 1. Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print("\n" + "=" * 80)
        print(f"📄 File saved to: {file_path}")

        # 2. Parse and save metadata
        parsed_metadata = None
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
                print(f"📝 Metadata received: {parsed_metadata}")
                with open(METADATA_FILE, "w", encoding="utf-8") as f:
                    json.dump(parsed_metadata, f, indent=4, ensure_ascii=False)
                print(f"💾 Metadata saved to {METADATA_FILE}")
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse metadata JSON: {e}")

        # ========================================
        # 3. Structural Analysis
        # ========================================
        print("\n" + "=" * 80)
        print("🔍 PHASE 1: Doc Analyzer (Structural Rules)")
        print("=" * 80)

        structural_result = doc_analyzer_agent.process_document(
            file_path,
            metadata=parsed_metadata,
        )

        # ========================================
        # 4. Contextual Analysis
        # ========================================
        contextual_result = None

        if theorist_agent:
            print("\n" + "=" * 80)
            print("🧠 PHASE 2: Theorist (Contextual Rules)")
            print("=" * 80)

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            parsed_json_path = os.path.join(
                OUTPUT_DIR,
                "parsed_docs",
                f"{base_name}_parsed.json",
            )

            if os.path.exists(parsed_json_path):
                print(f" 📂 Using parsed file: {parsed_json_path}")
                contextual_result = theorist_agent.analyze_document(
                    parsed_json_path=parsed_json_path,
                    metadata=parsed_metadata,
                )
            else:
                print(f" ⚠️ Parsed file not found: {parsed_json_path}")
                contextual_result = {
                    "error": "Parsed file not found for Theorist analysis"
                }
        else:
            print(" ⚠️ TheoristAgent not available, skipping contextual analysis")
            contextual_result = {"error": "TheoristAgent not initialized"}

        # ========================================
        # 5. Disclaimer & Glossary Analysis
        # ========================================
        dis_glos_result = None
        if dis_glos_agent and file.filename.lower().endswith(".pptx"):
            print("\n" + "=" * 80)
            print("⚖️ PHASE 3: Disclaimer & Glossary (new_dis_glos)")
            print("=" * 80)
            
            glossaires_path = os.path.join(CACHE_DIR, "glossaires.json")
            try:
                # The agent returns a full state dictionary, the report is in result["report"]
                agent_state = dis_glos_agent.run(file_path, glossaires_path)
                if agent_state and "report" in agent_state:
                    dis_glos_result = agent_state["report"]
                else:
                    dis_glos_result = {"error": "Agent execution failed or no report generated"}
            except Exception as e:
                print(f" ❌ Disclaimer agent failed: {e}")
                dis_glos_result = {"error": str(e)}
        elif not file.filename.lower().endswith(".pptx"):
            print(" ℹ️ Skipping phase 3 (only available for PPTX)")
        else:
            print(" ⚠️ DisGlosAgent not available")

        # ========================================
        # 6. Merge Results
        # ========================================
        combined_result = {
            "version": "v1.0.1_debug_glossary",
            "file_name": structural_result.get("file_name", file.filename),
            "document_id": structural_result.get("document_id"),
            "timestamp": structural_result.get("timestamp"),
            "analysis": structural_result.get("analysis", {}),
            "metrics": structural_result.get("metrics", {}),
            "doc_structure": structural_result.get("doc_structure", {}),
            "contextual_analysis": contextual_result,
            "disclaimer_analysis": dis_glos_result,
        }

        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE (Both Agents)")
        print("=" * 80)

        return combined_result

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/context")
async def get_context():
    return {
        "ephemeral": {
            "presentation": load_json(
                os.path.join(OUTPUT_DIR, "parsed_docs", "latest_presentation_parsed.json")
            ),
            "prospectus": load_json(
                os.path.join(OUTPUT_DIR, "parsed_docs", "latest_prospectus_parsed.json")
            ),
            "metadata": load_json(METADATA_FILE),
        },
        "persistent": {
            "regles_contextuelles": load_json(os.path.join(CACHE_DIR, "regles_contextuelles.json")),
            "regles_structurelles": load_json(os.path.join(CACHE_DIR, "regles_structurelles.json")),
            "glossaires": load_json(os.path.join(CACHE_DIR, "glossaires.json")),
            "funds": load_json(os.path.join(CACHE_DIR, "fond_registred.json")),
        },
    }


@app.post("/api/audit")
async def audit_documents(
    pptx_file: UploadFile = File(...),
    pdf_file: UploadFile = File(...)
):
    """
    Endpoint to trigger the Checker Memory Agent.
    Requires a PPTX and a PDF/DOCX file.
    Returns audit findings and global metrics.
    """
    global checker_agent
    
    try:
        # 1. Save uploaded files
        pptx_path = os.path.join(UPLOAD_DIR, pptx_file.filename)
        pdf_path = os.path.join(UPLOAD_DIR, pdf_file.filename)
        
        with open(pptx_path, "wb") as buffer:
            shutil.copyfileobj(pptx_file.file, buffer)
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(pdf_file.file, buffer)
            
        print(f"📄 Files saved: {pptx_path}, {pdf_path}")
        
        # 2. Initialize Checker Agent
        print("⏳ Initializing Checker Agent...")
        checker_agent = DocumentAgentWithMemory(
            pptx_path=pptx_path,
            pdf_path=pdf_path,
            questions_file=os.path.join(CACHE_DIR, "standard_questions.txt")
        )
        checker_agent.initialize()
        
        # 3. Run Audit
        print("🔍 Running Audit...")
        audit_results = checker_agent.run_audit()
        
        # 4. Generate Report (to get global metrics)
        report_path = os.path.join(OUTPUT_DIR, "QnA", "audit", "audit_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        report = checker_agent._generate_audit_report(report_path)
        
        return report

    except Exception as e:
        print(f"❌ Audit failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chatbot_endpoint(request: Request):
    global checker_agent
    data = await request.json()

    question = data.get("question")
    
    if not question:
        raise HTTPException(status_code=400, detail="Missing question")

    # Use Checker Agent if initialized
    if checker_agent:
        try:
            print(f"🤖 Checker Agent answering: {question}")
            response = checker_agent.answer_question(question)
            return response
        except Exception as e:
            print(f"❌ Checker Agent failed to answer: {e}")
            # Fallback to default logic below if needed, or raise error
            # For now, let's try to fallback or just return error
            pass

    # Fallback Logic (Original)
    ephemeral = data.get("ephemeral", {})
    persistent = data.get("persistent", {})

    system_prompt = """
Tu es un assistant de conformité STRICTEMENT RESTREINT AUX DONNÉES FOURNIES.
Tu ne peux ni inventer, ni compléter, ni utiliser des connaissances externes.
Si l'information n'existe pas dans les données fournies,
réponds exactement :
"Je ne dispose pas d'informations suffisantes dans les données fournies."
"""

    user_prompt = f"""
Voici la question utilisateur : {question}

--- CONTEXTE ÉPHÉMÈRE ---
{json.dumps(ephemeral, ensure_ascii=False, indent=2)}

--- CONTEXTE PERMANENT ---
{json.dumps(persistent, ensure_ascii=False, indent=2)}

Réponds en français.
"""

    answer = llm.generate_response(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=500,
    )

    if not answer:
        answer = "Je n'ai pas pu générer une réponse. Réessayez."

    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
