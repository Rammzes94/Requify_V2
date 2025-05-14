# ---------------------------------------------------------------------
# Section: Imports and Setup
# ---------------------------------------------------------------------
import os
import glob
import hashlib
import os
import sys
from dotenv import load_dotenv



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
import json

_00_utils.setup_project_directory()



# ---------------------------------------------------------------------
# Section: List Files in Directory and Filter Allowed File Types
# ---------------------------------------------------------------------

# Allowed file extensions for our application
allowed_extensions = ['.pdf', '.xlsx', '.xls', '.xlsm', '.pptx', '.docx', '.doc', '.png', '.jpg', '.jpeg', '.txt']
# Define the directory containing the files to check.
input_directory = '_01_input/raw'

files = []
for ext in allowed_extensions:
    # Extend the file list with glob results for each supported extension.
    files.extend(glob.glob(os.path.join(input_directory, '*' + ext)))

# Display the files found.
print("Supported files found:")
for file in files:
    print(file)


# ---------------------------------------------------------------------
# Section: Create a File Information List
# ---------------------------------------------------------------------
# For each file, gather filename, extension, size, and a content hash.
files_info = []
for file in files:
    size = os.path.getsize(file)
    # Compute a content hash using md5 over the entire file.
    # (Later we might use a "hash per page" for PDFs or more structured approaches.)
    with open(file, 'rb') as f:
        content = f.read()
    file_hash = hashlib.md5(content).hexdigest()
    
    files_info.append({
         'filename': file,
         'extension': os.path.splitext(file)[1].lower(),
         'size': size,
         'hash': file_hash
    })

# Print collected file info for debugging.
print("\nFile information collected:")
for info in files_info:
    print(info)

# ---------------------------------------------------------------------
# Section: Highest-Level pre AI Filterin - Compute Pairwise file_similarity
# ---------------------------------------------------------------------
# not: if two files have similar name AND size, they are similar for sure. They go to the AI to decide which one is newer


# We are just trying to filter out the obvious duplicate files.
# Without AI it is impossible to make sure a file is unique, so we dont try (based on file size / name etc.)
# two files have same hash = identical
# two files have the exact same size (single byte) = identical
# its ok if we dont fetch some things, we just want to use a little less compute on the AI part
# we will then not use the AI for the obvious duplicates
# no point in "identical in size" as it can be wrong (tested) and the hash already catches duplicates


# Group files by their extensions for comparison within the same file type.
files_by_extension = {}
for file_info in files_info:
    ext = file_info['extension']
    if ext not in files_by_extension:
        files_by_extension[ext] = []
    files_by_extension[ext].append(file_info)

similar_pairs = []

# Compare files only within the same file type.
import time
start_time = time.time()
total_comparisons = 0

for ext, files_group in files_by_extension.items():
    print(f"\nProcessing files with extension: {ext}")
    n = len(files_group)
    print(f"Found {n} files with extension {ext}")
    # Create a copy of the current group of files to allow modifications during iteration.
    filtered_files = files_group[:]
    for i in range(n):
        for j in range(i + 1, len(filtered_files)):
            total_comparisons += 1
            file1 = filtered_files[i]
            file2 = filtered_files[j]

            print(f"Comparing files:\n  File 1: {file1['filename']}\n  File 2: {file2['filename']}")
            
            # If the content hash is the same, files are identical.
            if file1['hash'] == file2['hash']:
                print("Files are identical based on hash")
                similar_pairs.append((file1['filename'], file2['filename']))
                filtered_files.pop(j)
                break

end_time = time.time()
print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
print(f"Total file comparisons made: {total_comparisons}")




# Update the files_by_extension dictionary with the filtered files.
filtered_files_by_extension = {}
for ext, files_group in files_by_extension.items():
    filtered_files_by_extension[ext] = [
        file for file in files_group if file['filename'] not in {pair[1] for pair in similar_pairs}
    ]

# Print the filtered files by extension for debugging.
print("\nFiltered files by extension:")
for ext, files in filtered_files_by_extension.items():
    print(f"Extension: {ext}")
    for file in files:
        print(f"  {file['filename']}")


# Save the filtered files by extension to a JSON file in the _03_output directory
# Ensure the _03_output directory exists
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
output_dir = os.path.join(project_root, "_03_output")
os.makedirs(output_dir, exist_ok=True)

output_json_path = os.path.join(output_dir, "filtered_files_by_extension.json")

with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(filtered_files_by_extension, json_file, indent=4)

print(f"\nFiltered files by extension have been saved to {output_json_path}")




print("\nSimilar files found:")
for pair in similar_pairs:
    print("File 1: {:<20s} File 2: {:<20s}".format(pair[0], pair[1]))

# Calculate and print the amounts
initial_files_count = len(files_info)
deleted_files_count = len(similar_pairs)

# Remaining files should be based on the full filtered list
remaining_files = [file for ext_files in filtered_files_by_extension.values() for file in ext_files]
remaining_files_count = len(remaining_files)

print("\nSummary:")
print(f"Initial files: {initial_files_count}")
print(f"Deleted files (similar pairs): {deleted_files_count}")
print(f"Remaining files: {remaining_files_count}")



# ---------------------------------------------------------------------
# Section: AI Similarity Check of everything that was not filtered out before
# ---------------------------------------------------------------------

# we will embed and compare page per page
# we should however create the embedding for each page and then just save it to a database.
# Then as next step we compare the pages based on the embeddings
# And then we start deleting things
# And then we already have the DB for our RAG system 
# We may want to use lightrag later, but it needs external db


