import os
import shutil
import sys
import asyncio
import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Ensure 'workflow' directory is in PYTHONPATH so internal imports (like 'from config import...') work
sys.path.append(os.path.join(os.path.dirname(__file__), "workflow"))

# Import both agents
from workflow.doc_analyzer import DocumentComplianceAgent
from workflow.theorist import TheoristAgent

app = FastAPI(title="Compliance Check API")

# Configure CORS - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "workflow", "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve uploaded files statically so frontend can access them
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Initialize both Agents at startup
print("⏳ Initializing AI Agents (loading models)...")
doc_analyzer_agent = None
theorist_agent = None

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

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Compliance Backend is running (2 agents)"}

@app.post("/api/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """
    Endpoint to receive a document and trigger the compliance analysis workflow.
    Runs BOTH doc_analyzer (structural) and theorist (contextual) agents.
    Optionally accepts metadata as JSON string for rule filtering.
    """
    if not doc_analyzer_agent:
        raise HTTPException(status_code=503, detail="DocumentComplianceAgent not initialized.")

    try:
        # 1. Save the uploaded file temporarily
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"\n{'='*80}")
        print(f"📄 File saved to: {file_path}")

        # 2. Parse metadata if provided
        parsed_metadata = None
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
                print(f"📝 Metadata received: {parsed_metadata}")
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse metadata JSON: {e}")

        # ========================================
        # 3. Run DocAnalyzer (Structural Analysis)
        # ========================================
        print(f"\n{'='*80}")
        print("🔍 PHASE 1: Doc Analyzer (Structural Rules)")
        print("="*80)
        
        structural_result = doc_analyzer_agent.process_document(file_path, metadata=parsed_metadata)
        
        # ========================================
        # 4. Run Theorist (Contextual Analysis)
        # ========================================
        contextual_result = None
        if theorist_agent:
            print(f"\n{'='*80}")
            print("🧠 PHASE 2: Theorist (Contextual Rules)")
            print("="*80)
            
            # Find the parsed JSON file created by doc_analyzer
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            parsed_json_path = os.path.join(OUTPUT_DIR, "parsed_docs", f"{base_name}_parsed.json")
            
            if os.path.exists(parsed_json_path):
                print(f"  📂 Using parsed file: {parsed_json_path}")
                contextual_result = theorist_agent.analyze_document(
                    parsed_json_path=parsed_json_path,
                    metadata=parsed_metadata
                )
            else:
                print(f"  ⚠️ Parsed file not found: {parsed_json_path}")
                contextual_result = {"error": "Parsed file not found for Theorist analysis"}
        else:
            print("  ⚠️ TheoristAgent not available, skipping contextual analysis")
            contextual_result = {"error": "TheoristAgent not initialized"}

        # ========================================
        # 5. Merge Results
        # ========================================
        combined_result = {
            "file_name": structural_result.get("file_name", file.filename),
            "document_id": structural_result.get("document_id"),
            "timestamp": structural_result.get("timestamp"),
            
            # Structural analysis (from doc_analyzer)
            "analysis": structural_result.get("analysis", {}),
            "metrics": structural_result.get("metrics", {}),
            "doc_structure": structural_result.get("doc_structure", {}),
            
            # Contextual analysis (from theorist)
            "contextual_analysis": contextual_result,
        }
        
        print(f"\n{'='*80}")
        print("✅ ANALYSIS COMPLETE (Both Agents)")
        print("="*80)
        
        return combined_result

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup if desired
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
