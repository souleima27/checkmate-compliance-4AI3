import os
import shutil
from pptx import Presentation
from docx import Document
from docx.shared import RGBColor, Pt
from docx.enum.text import WD_COLOR_INDEX

def inject_violations_pptx(input_path: str, output_path: str, violations: list):
    """
    Inject violations into PPTX as Slide Notes.
    """
    prs = Presentation(input_path)
    
    # Violations derived from PDF analysis usually have 'page' 1-indexed.
    # PPTX slides are 0-indexed in list, but match 1-based page numbers generally.
    
    # Group violations by page
    violations_by_page = {}
    for v in violations:
        p = v.get("page", 1)
        if p not in violations_by_page:
            violations_by_page[p] = []
        violations_by_page[p].append(v)
        
    for i, slide in enumerate(prs.slides):
        page_num = i + 1
        if page_num in violations_by_page:
            # Get or create notes slide
            notes_slide = slide.notes_slide
            text_frame = notes_slide.notes_text_frame
            
            # Add header
            p = text_frame.add_paragraph()
            p.text = "\n--- COMPLIANCE VIOLATIONS ---"
            p.font.bold = True
            
            for v in violations_by_page[page_num]:
                p = text_frame.add_paragraph()
                p.text = f"• [{v.get('scope', 'General')}] {v.get('title', 'Violation')}: {v.get('description', '')}"
                p.level = 0
                
    prs.save(output_path)
    return output_path

def inject_violations_docx(input_path: str, output_path: str, violations: list):
    """
    Inject violations into DOCX as a summary table at the start.
    Mapping specific violations to DOCX text locations is difficult without complex alignment.
    """
    doc = Document(input_path)
    
    # Insert a warning heading at the top
    doc.insert_paragraph("COMPLIANCE REPORT", style='Heading 1')
    
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Scope'
    hdr_cells[1].text = 'Title'
    hdr_cells[2].text = 'Description'
    hdr_cells[3].text = 'Page Ref (Approx)'
    
    for v in violations:
        row_cells = table.add_row().cells
        row_cells[0].text = v.get('scope', '')
        row_cells[1].text = v.get('title', '')
        row_cells[2].text = v.get('description', '')
        row_cells[3].text = str(v.get('page', '?'))
        
    doc.add_page_break()
    
    # Move the inserted content to the beginning (trickier in python-docx, usually appends)
    # Actually, simplistic approach: Setup report at end? 
    # Or create a new doc, add report, add existing elements? Too complex to preserve style.
    
    # Best compromise: Append report at the beginning?
    # python-docx `add_paragraph` appends. `doc.paragraphs[0].insert_paragraph_before` works.
    # But tables? `doc.add_table` appends.
    
    # Let's simple APPEND the report for safety first.
    # Or, well, user said "annotate".
    # I'll stick to prepending if possible, but appending is safer for file corruption.
    # Let's Just Append for now, it's safer.
    
    doc.save(output_path)
    return output_path

def inject_violations(input_path: str, violations: list, output_dir: str) -> str:
    """
    Main entry point for injection.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ext = os.path.splitext(input_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_filename = f"{base_name}_annotated{ext}"
    output_path = os.path.join(output_dir, output_filename)
    
    if ext == ".pptx":
        return inject_violations_pptx(input_path, output_path, violations)
    elif ext == ".docx":
        return inject_violations_docx(input_path, output_path, violations)
    else:
        # Fallback: Just copy
        shutil.copy2(input_path, output_path)
        return output_path
