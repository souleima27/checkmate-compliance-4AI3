import os
import shutil
import zipfile
import re
import json
from pathlib import Path
from datetime import datetime
from lxml import etree
from collections import defaultdict
from pptx import Presentation
from docx import Document

# ==============================================================================
# 1. CLASS MANAGER FOR XML COMMENTS
# ==============================================================================

class PowerPointCommentManager:
    """Manages low-level XML injection for PowerPoint comments."""
    
    def __init__(self, extract_dir, author_name="Compliance Bot", author_initials="CB"):
        self.extract_dir = Path(extract_dir)
        self.author_name = author_name
        self.author_initials = author_initials
        self.author_id = 1
        self.comment_counter = 1
        
        # XML Namespaces
        self.namespaces = {
            'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
            'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
            'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
        }
        
        # Register namespaces
        for prefix, uri in self.namespaces.items():
            etree.register_namespace(prefix, uri)
    
    def ensure_authors_file(self):
        """Ensures commentAuthors.xml exists."""
        authors_path = self.extract_dir / 'ppt' / 'commentAuthors.xml'
        
        if not authors_path.exists():
            print(f"DEBUG: Creating {authors_path}")
            authors_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:cmAuthorLst xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cmAuthor id="{self.author_id}" name="{self.author_name}" initials="{self.author_initials}" lastIdx="0" clrIdx="0"/>
</p:cmAuthorLst>'''
            
            authors_path.parent.mkdir(parents=True, exist_ok=True)
            with open(authors_path, 'w', encoding='utf-8') as f:
                f.write(authors_xml)
            
            self._add_authors_relationship()
    
    def _add_authors_relationship(self):
        """Adds relationship to commentAuthors.xml in presentation.xml.rels."""
        rels_path = self.extract_dir / 'ppt' / '_rels' / 'presentation.xml.rels'
        
        if rels_path.exists():
            tree = etree.parse(str(rels_path))
            root = tree.getroot()
            
            exists = False
            for rel in root.findall('.//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship'):
                if rel.get('Target') == 'commentAuthors.xml':
                    exists = True
                    break
            
            if not exists:
                max_id = 0
                for rel in root.findall('.//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship'):
                    rel_id = rel.get('Id', '')
                    if rel_id.startswith('rId'):
                        try:
                            num = int(rel_id[3:])
                            max_id = max(max_id, num)
                        except ValueError:
                            pass
                
                new_id = f'rId{max_id + 1}'
                rel_elem = etree.SubElement(
                    root,
                    '{http://schemas.openxmlformats.org/package/2006/relationships}Relationship'
                )
                rel_elem.set('Id', new_id)
                rel_elem.set('Type', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/commentAuthors')
                rel_elem.set('Target', 'commentAuthors.xml')
                
                tree.write(str(rels_path), encoding='utf-8', xml_declaration=True)
                print("DEBUG: Added authors relationship")
    
    def add_comment_to_slide(self, slide_number, comment_text, position=(914400, 914400)):
        """Adds a comment to a specific slide."""
        # Note: slide_number is 1-based index from file naming (slide1.xml)
        comments_path = self.extract_dir / 'ppt' / 'comments' / f'comment{slide_number}.xml'
        slide_path = self.extract_dir / 'ppt' / 'slides' / f'slide{slide_number}.xml'
        
        if not slide_path.exists():
            print(f"⚠️ Slide {slide_number} not found at {slide_path}")
            return False
        
        comments_path.parent.mkdir(parents=True, exist_ok=True)
        
        if comments_path.exists():
            tree = etree.parse(str(comments_path))
            root = tree.getroot()
        else:
            root = etree.Element('{http://schemas.openxmlformats.org/presentationml/2006/main}cmLst')
        
        comment_id = self.comment_counter
        self.comment_counter += 1
        
        cm = etree.SubElement(
            root,
            '{http://schemas.openxmlformats.org/presentationml/2006/main}cm',
            authorId=str(self.author_id),
            dt=datetime.now().isoformat(),
            idx=str(comment_id)
        )
        
        pos = etree.SubElement(
            cm,
            '{http://schemas.openxmlformats.org/presentationml/2006/main}pos',
            x=str(position[0]),
            y=str(position[1])
        )
        
        text = etree.SubElement(
            cm,
            '{http://schemas.openxmlformats.org/presentationml/2006/main}text'
        )
        text.text = comment_text
        
        tree = etree.ElementTree(root)
        tree.write(str(comments_path), encoding='utf-8', xml_declaration=True, pretty_print=True)
        
        self._add_comment_relationship(slide_number)
        print(f"DEBUG: Added comment {comment_id} to slide {slide_number} at {position}")
        return True
    
    def _add_comment_relationship(self, slide_number):
        """Adds relationship to the comment file in slide rels."""
        rels_path = self.extract_dir / 'ppt' / 'slides' / '_rels' / f'slide{slide_number}.xml.rels'
        rels_path.parent.mkdir(parents=True, exist_ok=True)
        
        if rels_path.exists():
            tree = etree.parse(str(rels_path))
            root = tree.getroot()
        else:
            root = etree.Element('{http://schemas.openxmlformats.org/package/2006/relationships}Relationships')
        
        exists = False
        target_file = f'../comments/comment{slide_number}.xml'
        
        for rel in root.findall('.//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship'):
            if 'comments' in rel.get('Type', '') and target_file in rel.get('Target', ''):
                exists = True
                break
        
        if not exists:
            max_id = 0
            for rel in root.findall('.//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship'):
                rel_id = rel.get('Id', '')
                if rel_id.startswith('rId'):
                    try:
                        num = int(rel_id[3:])
                        max_id = max(max_id, num)
                    except ValueError:
                        pass
            
            new_id = f'rId{max_id + 1}'
            rel_elem = etree.SubElement(
                root,
                '{http://schemas.openxmlformats.org/package/2006/relationships}Relationship'
            )
            rel_elem.set('Id', new_id)
            rel_elem.set('Type', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments')
            rel_elem.set('Target', target_file)
            
            tree = etree.ElementTree(root)
            tree.write(str(rels_path), encoding='utf-8', xml_declaration=True, pretty_print=True)

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def unpack_pptx(pptx_path, extract_dir):
    print(f"DEBUG: Unpacking PPTX to {extract_dir}")
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(pptx_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

def pack_pptx(extract_dir, output_path):
    print(f"DEBUG: Packing PPTX to {output_path}")
    if os.path.exists(output_path):
        os.remove(output_path)
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, extract_dir)
                zipf.write(file_path, arcname)

def format_comment_text(violation):
    """Formats the violation dictionary into a readable string."""
    lines = []
    
    # Severity Emoji
    severity = violation.get('severity', 'UNKNOWN').upper()
    emoji = {'CRITICAL': '🔴', 'WARNING': '🟡', 'INFO': '🔵'}.get(severity, '⚠️')
    
    title = violation.get('title', 'Problème de conformité')
    lines.append(f"{emoji} {title}")
    lines.append(f"Scope: {violation.get('scope', 'General')}")
    lines.append("")
    
    # Description
    desc = violation.get('description', '')
    if desc:
        lines.append(desc)
    
    # Suggestion
    fix = violation.get('suggested_fix', '') or violation.get('recommendation', '')
    if fix:
        lines.append("")
        lines.append(f"✅ Suggestion: {fix}")
        
    return "\n".join(lines)

def extract_shape_id_from_element_id(element_id):
    """
    Parses shape index from 'slide_X_shape_Y' or 'Shape Y'.
    Returns 0-based index (int) or None.
    """
    if not element_id:
        return None
        
    s_id = str(element_id)
    
    # 1. Try 'slide_X_shape_Y' format (produced by doc_analyzer)
    # This is 1-based index
    match = re.search(r'shape_(\d+)', s_id, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1)) - 1  # Convert 1-based to 0-based
        except ValueError:
            pass

    # 2. Try 'Shape Y' format
    clean = s_id.replace("Shape", "").strip()
    match = re.search(r'(\d+)', clean)
    if match:
        try:
            val = int(match.group(1))
            # Heuristic: if value is small, assume it's an index. 
            # If it comes from 'Shape 4', it might be 1-based (PowerPoint UI name) or 0-based.
            # Let's assume 1-based for safety if ambiguous, or just use as is?
            # doc_analyzer is the main source, handled above.
            return val
        except ValueError:
            pass
            
    return None

def get_shape_position_in_slide(extract_dir, slide_num, shape_index):
    """
    Finds shape by 0-based index in the slide XML.
    Returns (x, y) in EMUs or None.
    """
    try:
        slide_path = Path(extract_dir) / 'ppt' / 'slides' / f'slide{slide_num}.xml'
        tree = etree.parse(str(slide_path))
        root = tree.getroot()
        
        namespaces = {
            'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
            'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'
        }
        
        shapes = root.findall('.//p:sp', namespaces)
        
        if shape_index is not None and 0 <= shape_index < len(shapes):
            shape = shapes[shape_index]
            xfrm = shape.find('.//p:spPr/a:xfrm/a:off', namespaces)
            if xfrm is not None:
                x = xfrm.get('x')
                y = xfrm.get('y')
                if x and y:
                    return (int(x), int(y))
        else:
            print(f"DEBUG: Shape index {shape_index} out of bounds (found {len(shapes)} shapes)")
            
    except Exception as e:
        print(f"Error finding shape pos: {e}")
    return None


# ==============================================================================
# 3. MAIN INJECTION ENTRY POINTS
# ==============================================================================

def inject_violations_pptx(input_path: str, output_path: str, violations: list):
    """
    Injects violations as NATIVE COMMENTS into the PPTX.
    """
    print(f"DEBUG: Starting PPTX injection with {len(violations)} violations")
    import tempfile
    
    # 1. Prepare temp directory
    extract_dir = tempfile.mkdtemp(prefix='pptx_inject_')
    
    try:
        # 2. Unpack
        unpack_pptx(input_path, extract_dir)
        
        # 3. Setup Manager
        manager = PowerPointCommentManager(extract_dir)
        manager.ensure_authors_file()
        
        # 4. Group violations
        # Group by dictionary key (slide_num, element_id) -> list of violations
        grouped = defaultdict(list)
        
        for v in violations:
            # Assume 'page' corresponds to slide number (1-based)
            slide_num = v.get('page', 1)
            # 'element_id' might be "Shape 5" or None
            element_id = v.get('element_id') 
            grouped[(slide_num, element_id)].append(v)
            
        print(f"DEBUG: Grouped violations into {len(grouped)} locations")

        # 5. Inject
        default_positions = {} # per slide
        
        for (slide_num, element_id), v_list in grouped.items():
            
            # Find position
            position = None
            shape_idx = extract_shape_id_from_element_id(element_id)
            
            if shape_idx is not None:
                position = get_shape_position_in_slide(extract_dir, slide_num, shape_idx)
                if position:
                     print(f"DEBUG: Found shape position for slide {slide_num}, shape {shape_idx}: {position}")
            
            # Fallback position logic
            if not position:
                if slide_num not in default_positions:
                    default_positions[slide_num] = 914400 # 1 inch
                
                # Stack them vertically on the left/top
                y_pos = default_positions[slide_num]
                position = (457200, y_pos) # 0.5 inch x, shifting y
                default_positions[slide_num] += 1500000 # increment ~1.5 inch
                print(f"DEBUG: Using default position for slide {slide_num}: {position}")
            else:
                 # Shift slightly right/down so it doesn't obscure the shape completely
                 position = (position[0] + 200000, position[1] + 200000)
            
            # Create comment text (combine multiple if needed, or add separate)
            for v in v_list:
                text = format_comment_text(v)
                manager.add_comment_to_slide(slide_num, text, position)
        
        # 6. Repack
        pack_pptx(extract_dir, output_path)
        print("DEBUG: Injection completed successfully")
        return output_path
        
    except Exception as e:
        print(f"❌ PPTX Injection failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to copy if XML manipulation fails
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        shutil.copy2(input_path, output_path)
        return output_path
    finally:
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)


def inject_violations_docx(input_path: str, output_path: str, violations: list):
    """
    Inject violations into DOCX as a summary table (Existing logic).
    """
    doc = Document(input_path)
    
    doc.insert_paragraph("COMPLIANCE REPORT", style='Heading 1')
    
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Scope'
    hdr_cells[1].text = 'Title'
    hdr_cells[2].text = 'Description'
    hdr_cells[3].text = 'Page Ref'
    
    for v in violations:
        row_cells = table.add_row().cells
        row_cells[0].text = v.get('scope', '')
        row_cells[1].text = v.get('title', '')
        row_cells[2].text = v.get('description', '')
        row_cells[3].text = str(v.get('page', '?'))
        
    doc.save(output_path)
    return output_path


def inject_violations(input_path: str, violations: list, output_dir: str) -> str:
    """
    Main entry point for injection.
    """
    print(f"DEBUG: Processing injection for {input_path}")
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
        shutil.copy2(input_path, output_path)
        return output_path

