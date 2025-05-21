import os
import win32com.client

# -------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------
INPUT_DIR = "input/raw"
OUTPUT_DIR = "input/processed"
# Example input file within the raw directory (adjust as needed)
EXAMPLE_INPUT_FILE = os.path.join(INPUT_DIR, "sample_for_comparer", "demo.pptx")


# -------------------------------------------------------------------------------------
# Path Configuration
# -------------------------------------------------------------------------------------
# Use the example file for demonstration, ensure it exists or change it
input_file = EXAMPLE_INPUT_FILE # Change this to test other files if needed

# Get absolute paths
input_path = os.path.abspath(input_file)
output_dir_abs = os.path.abspath(OUTPUT_DIR) # Use the constant
base_name = os.path.splitext(os.path.basename(input_path))[0]
output_path = os.path.join(output_dir_abs, base_name + ".pdf") # Use absolute output dir

# Debug prints
print(f"Input path: {input_path}")
print(f"Output dir: {output_dir_abs}")
print(f"Output path: {output_path}")

# Ensure output directory exists
if not os.path.exists(output_dir_abs):
    os.makedirs(output_dir_abs)

# Ensure the input file exists
if not os.path.exists(input_path):
    raise FileNotFoundError(f"❌ File does not exist: {input_path}")

def convert_word_to_pdf(input_path, output_path):
    print("🔄 Converting Word document...")
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(input_path)
        doc.SaveAs(output_path, FileFormat=17)  # 17 = PDF
        doc.Close()
        print(f"✅ Word to PDF: {output_path}")
    except Exception as e:
        print(f"❌ Word conversion error: {e}")
    finally:
        word.Quit()

def convert_powerpoint_to_pdf(input_path, output_path):
    print("🔄 Converting PowerPoint presentation...")
    ppt = win32com.client.Dispatch("PowerPoint.Application")
    # Don't set ppt.Visible = False — causes error
    try:
        presentation = ppt.Presentations.Open(input_path, WithWindow=False)
        presentation.SaveAs(output_path, FileFormat=32)  # 32 = PDF
        presentation.Close()
        print(f"✅ PowerPoint to PDF: {output_path}")
    except Exception as e:
        print(f"❌ PowerPoint conversion error: {e}")
    finally:
        ppt.Quit()

# Decide which converter to use
ext = os.path.splitext(input_path)[1].lower()

if ext in [".doc", ".docx"]:
    convert_word_to_pdf(input_path, output_path)
elif ext in [".ppt", ".pptx"]:
    convert_powerpoint_to_pdf(input_path, output_path)
else:
    print(f"❌ Unsupported file type: {ext}")
