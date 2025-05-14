# This script works, but its not the best practice. 
# Its not a good idea to scan a huge dir and decide on the latest file just 
# based on the modified time. We will use embeddings instead.

import datetime
import difflib
import json  # For JSON serialization of outputs
from pathlib import Path
from typing import List, Dict, Tuple

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool


# ---------------------------------------------------------------------
# Section: Setup and Imports
# Description: Import required modules, load environment variables,
#              add parent directory, and set up the project directory.
# ---------------------------------------------------------------------
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
_00_utils.setup_project_directory()
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------
# Section: Global Configuration
# Description: Set the target directory to analyze.
# ---------------------------------------------------------------------
TARGET_DIR = "_01_input/raw"

# ---------------------------------------------------------------------
# Section: File Scanning and Similarity Tools with Debug Prints
# Description: Define tools with additional prints to log calls and outputs.
#              Each tool returns a JSON-formatted string.
# ---------------------------------------------------------------------
@tool(show_result=True)
def scan_directory(directory_path: str) -> str:
    """
    Scan a directory and return information about all files.
    """
    print(f"[DEBUG] scan_directory called with directory_path: {directory_path}")
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        err = [{"error": f"Directory {directory_path} does not exist or is not a directory"}]
        print(f"[DEBUG] scan_directory error: {err}")
        return json.dumps(err, indent=2)
    
    files_info = []
    for file_path in directory.glob('*'):
        if file_path.is_file():
            stats = file_path.stat()
            file_data = {
                "path": str(file_path),
                "name": file_path.name,
                "extension": file_path.suffix,
                "size_bytes": stats.st_size,
                "created_time": datetime.datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "modified_time": datetime.datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "accessed_time": datetime.datetime.fromtimestamp(stats.st_atime).isoformat(),
            }
            files_info.append(file_data)
    print(f"[DEBUG] scan_directory found {len(files_info)} files.")
    return json.dumps(files_info, indent=2)

@tool(show_result=True)
def find_similar_files(files_info_json: str, similarity_threshold: float = 0.7) -> str:
    """
    Find groups of similar files based on their names.
    """
    print(f"[DEBUG] find_similar_files called with similarity_threshold: {similarity_threshold}")
    files_info = json.loads(files_info_json)
    files = [(info["name"], info) for info in files_info]
    similar_groups = []
    processed = set()
    
    for i, (file_name, file_info) in enumerate(files):
        if i in processed:
            continue
        group = [file_info]
        processed.add(i)
        for j, (other_name, other_info) in enumerate(files):
            if j in processed or i == j:
                continue
            similarity = difflib.SequenceMatcher(None, file_name, other_name).ratio()
            if similarity >= similarity_threshold:
                group.append(other_info)
                processed.add(j)
        if len(group) > 1:
            similar_groups.append(group)
    print(f"[DEBUG] find_similar_files found {len(similar_groups)} groups.")
    return json.dumps(similar_groups, indent=2)

@tool(show_result=True)
def determine_newer_file(file_group_json: str) -> str:
    """
    Determine which file in a group is likely more up-to-date.
    """
    print("[DEBUG] determine_newer_file called.")
    file_group = json.loads(file_group_json)
    
    if not file_group or len(file_group) < 2:
        error_msg = {"error": "Need at least two files to compare"}
        print(f"[DEBUG] determine_newer_file error: {error_msg}")
        return json.dumps(error_msg, indent=2)
    
    sorted_by_time = sorted(file_group, key=lambda x: x["modified_time"], reverse=True)
    sorted_by_size = sorted(file_group, key=lambda x: x["size_bytes"], reverse=True)
    newest_file = sorted_by_time[0]
    largest_file = sorted_by_size[0]
    
    result = {
        "files_compared": len(file_group),
        "newest_file": {
            "name": newest_file["name"],
            "path": newest_file["path"],
            "modified_time": newest_file["modified_time"],
            "size_bytes": newest_file["size_bytes"]
        },
        "largest_file": {
            "name": largest_file["name"],
            "path": largest_file["path"],
            "modified_time": largest_file["modified_time"],
            "size_bytes": largest_file["size_bytes"]
        },
        "is_newest_also_largest": newest_file["path"] == largest_file["path"],
        "all_files": [
            {
                "name": f["name"],
                "path": f["path"],
                "modified_time": f["modified_time"],
                "size_bytes": f["size_bytes"]
            } for f in file_group
        ]
    }
    
    if not result["is_newest_also_largest"]:
        time_diff = datetime.datetime.fromisoformat(newest_file["modified_time"]) - \
                    datetime.datetime.fromisoformat(largest_file["modified_time"])
        size_ratio = largest_file["size_bytes"] / max(newest_file["size_bytes"], 1)
        result["conflict_analysis"] = {
            "time_difference_seconds": abs(time_diff.total_seconds()),
            "size_ratio": size_ratio,
            "recommendation": "newest" if abs(time_diff.total_seconds()) > 3600 else "largest" if size_ratio > 1.5 else "manual_review"
        }
    print("[DEBUG] determine_newer_file completed analysis.")
    return json.dumps(result, indent=2)

@tool(show_result=True)
def compare_file_contents(file_path1: str, file_path2: str) -> str:
    """
    Compare the contents of two files.
    """
    print(f"[DEBUG] compare_file_contents called for files: {file_path1} and {file_path2}")
    path1 = Path(file_path1)
    path2 = Path(file_path2)
    
    if not path1.exists() or not path2.exists():
        err = {"error": "One or both files do not exist"}
        print(f"[DEBUG] compare_file_contents error: {err}")
        return json.dumps(err, indent=2)
    
    try:
        with open(path1, 'r', encoding='utf-8') as f1, open(path2, 'r', encoding='utf-8') as f2:
            content1 = f1.readlines()
            content2 = f2.readlines()
            
        matcher = difflib.SequenceMatcher(None, ''.join(content1), ''.join(content2))
        similarity_ratio = matcher.ratio()
        diff = list(difflib.unified_diff(
            content1, content2, 
            fromfile=str(path1), 
            tofile=str(path2),
            n=3
        ))
        
        result = {
            "similarity_ratio": similarity_ratio,
            "different_lines": len(diff) - 6 if len(diff) > 6 else 0,
            "is_identical": similarity_ratio > 0.99,
            "diff_preview": ''.join(diff[:20]) if diff else "Files are identical",
        }
        print(f"[DEBUG] compare_file_contents result: Similarity = {similarity_ratio:.2f}")
        return json.dumps(result, indent=2)
    except UnicodeDecodeError:
        size1 = path1.stat().st_size
        size2 = path2.stat().st_size
        result = {
            "error": "Cannot compare binary files line by line",
            "file1_size": size1,
            "file2_size": size2,
            "size_difference": abs(size1 - size2),
            "size_ratio": max(size1, size2) / max(min(size1, size2), 1)
        }
        print(f"[DEBUG] compare_file_contents (binary) result: {result}")
        return json.dumps(result, indent=2)
    except Exception as e:
        err = {"error": f"Error comparing files: {str(e)}"}
        print(f"[DEBUG] compare_file_contents exception: {err}")
        return json.dumps(err, indent=2)

# ---------------------------------------------------------------------
# Section: Agent Configuration using Sample Agent Syntax
# Description: Instantiate the agent using the sample syntax.
# ---------------------------------------------------------------------
# Create a model instance.
model = OpenAIChat(id="gpt-4o-mini")

file_comparison_agent = Agent(
    model=model,  
    markdown=True,
    debug_mode=True,
    description="You are an agent that analyzes files in a directory to determine which ones are similar and which file is likely the most up-to-date. You consider file metadata such as modified time, file size, and content differences. Provide a detailed explanation of your reasoning along with the functions called and their key outputs.",
    tools=[scan_directory, find_similar_files, determine_newer_file, compare_file_contents]
)

# ---------------------------------------------------------------------
# Section: Running the Agent with Debugging
# Description: Use the agent to process the task by providing a detailed prompt.
# ---------------------------------------------------------------------
print("[DEBUG] Starting agent run with prompt...")
prompt = f"""
Analyze the files in the directory "{TARGET_DIR}".
Please:
1. Scan the directory to extract all file metadata.
2. Find groups of similar files based on file name similarity.
3. For each group, determine which file is likely the most up-to-date based on modification time, file size, and (if applicable) content differences.
Provide a detailed explanation of your reasoning, listing which functions are called and the key outputs of each step.
"""

agent_result = file_comparison_agent.run(prompt)

print("\n[DEBUG] Agent Analysis Result:")
print(agent_result)
