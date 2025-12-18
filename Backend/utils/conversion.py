import os
import subprocess
import shutil
import platform
import sys

def get_soffice_path():
    """Find the path to LibreOffice soffice executable."""
    # 1. Check PATH
    path = shutil.which("soffice")
    if path: return path
    
    # 2. Check Windows Defaults
    if platform.system() == "Windows":
        candidates = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            # Add more potential paths if needed
            os.path.expandvars(r"%PROGRAMFILES%\LibreOffice\program\soffice.exe"),
            os.path.expandvars(r"%PROGRAMFILES(X86)%\LibreOffice\program\soffice.exe"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
    return None

def convert_to_pdf(input_path: str, output_dir: str) -> str:
    """
    Converts input file (pptx, docx) to PDF in the output_dir.
    Returns the path to the generated PDF.
    """
    # Resolve absolute paths
    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    soffice = get_soffice_path()
    
    if not soffice:
        # Fallback suggestion
        raise FileNotFoundError("LibreOffice not found! Please install LibreOffice (https://www.libreoffice.org/).")
        
    # Command: soffice --headless --convert-to pdf --outdir <out> <in>
    cmd = [
        soffice,
        "--headless",
        "--convert-to", "pdf",
        "--outdir", output_dir,
        input_path
    ]
    
    print(f"Running conversion: {cmd}")
    
    # Run conversion
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Conversion Error Stdout: {e.stdout}")
        print(f"Conversion Error Stderr: {e.stderr}")
        raise RuntimeError(f"LibreOffice conversion failed: {e.stderr}")
        
    # Determine output filename
    # LibreOffice saves as <filename>.pdf
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_pdf = os.path.join(output_dir, base_name + ".pdf")
    
    # Verify existence
    if not os.path.exists(output_pdf):
        # Sometimes casing issues or weird output names?
        raise FileNotFoundError(f"PDF not created at {output_pdf}")
        
    return output_pdf
