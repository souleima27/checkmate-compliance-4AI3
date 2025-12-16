
import os
import sys
# Make sure we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.conversion import convert_to_pdf, get_soffice_path

def test_conversion():
    print("--- Diagnostic Start ---")
    
    # 1. Check Path
    path = get_soffice_path()
    print(f"Detected LibreOffice Path: {path}")
    
    if not path:
        print("ERROR: LibreOffice not found via get_soffice_path()")
        # Try to find it manually to help
        candidates = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        ]
        for c in candidates:
            exists = os.path.exists(c)
            print(f"Checking {c}: {'FOUND' if exists else 'not found'}")
        return

    # 2. Create Dummy File
    input_file = "test_doc.docx"
    with open(input_file, "w") as f:
        f.write("Test content")
    print(f"Created dummy file: {os.path.abspath(input_file)}")
    
    # 3. Try Conversion
    output_dir = "test_output"
    try:
        pdf = convert_to_pdf(input_file, output_dir)
        print(f"SUCCESS: Created {pdf}")
    except Exception as e:
        print(f"FAILURE: {e}")

if __name__ == "__main__":
    test_conversion()
