#!/usr/bin/env python3
"""
results_reporter.py

This script generates HTML reports from test results.
It looks for the latest test results and generates a comprehensive HTML report.
"""

import os
import sys
import json
import glob
from datetime import datetime
import logging

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src import _00_utils
_00_utils.setup_project_directory()

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logger = _00_utils.get_logger("TestReporter")

# Constants
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                          "output", "test_results")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

def get_latest_test_results():
    """Get the latest test result files"""
    # Ensure directories exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Find all JSON result files
    json_files = glob.glob(os.path.join(RESULTS_DIR, "e2e_test_results_*.json"))
    
    if not json_files:
        logger.warning("No test result files found", extra={"icon": "⚠️"})
        return None
    
    # Sort by timestamp (newest first)
    json_files.sort(reverse=True)
    
    latest_results = []
    
    # Group results by run (same timestamp)
    current_timestamp = None
    
    for file_path in json_files:
        timestamp = os.path.basename(file_path).replace("e2e_test_results_", "").replace(".json", "")
        
        if current_timestamp is None:
            current_timestamp = timestamp
        
        if timestamp == current_timestamp:
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                    latest_results.append(result)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}", extra={"icon": "❌"})
        else:
            # Stop after getting all results from latest run
            break
    
    return latest_results if latest_results else None

def generate_html_report(results):
    """Generate an HTML report from test results"""
    if not results:
        logger.warning("No test results to report", extra={"icon": "⚠️"})
        return False
    
    # Create timestamp for the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(REPORTS_DIR, f"test_report_{timestamp}.html")
    
    # Sort results by scenario ID
    results.sort(key=lambda x: x.get("scenario_id", 0))
    
    # Count passes and failures
    passed = sum(1 for r in results if r.get("passed", False))
    failed = len(results) - passed
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Requify Test Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            .header {{
                background-color: #f8f8f8;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
                border-left: 5px solid #2c3e50;
            }}
            .summary {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 30px;
            }}
            .metric {{
                text-align: center;
                padding: 15px;
                border-radius: 5px;
                flex: 1;
                margin: 0 10px;
            }}
            .metric.total {{
                background-color: #e9f7fe;
                border-left: 4px solid #3498db;
            }}
            .metric.passed {{
                background-color: #e6ffed;
                border-left: 4px solid #2ecc71;
            }}
            .metric.failed {{
                background-color: #fff0f0;
                border-left: 4px solid #e74c3c;
            }}
            .metric h2 {{
                font-size: 32px;
                margin: 5px 0;
            }}
            .scenario {{
                margin-bottom: 20px;
                border-radius: 5px;
                overflow: hidden;
            }}
            .scenario-header {{
                padding: 12px 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .passed-header {{
                background-color: #2ecc71;
                color: white;
            }}
            .failed-header {{
                background-color: #e74c3c;
                color: white;
            }}
            .scenario-body {{
                padding: 15px;
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-top: none;
            }}
            .steps {{
                margin-top: 10px;
            }}
            .timestamp {{
                font-size: 14px;
                color: #888;
                margin-top: 5px;
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
                font-size: 14px;
                color: #888;
                padding-top: 20px;
                border-top: 1px solid #eee;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Requify Document Processing Test Report</h1>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="summary">
            <div class="metric total">
                <p>Total Scenarios</p>
                <h2>{len(results)}</h2>
            </div>
            <div class="metric passed">
                <p>Passed</p>
                <h2>{passed}</h2>
            </div>
            <div class="metric failed">
                <p>Failed</p>
                <h2>{failed}</h2>
            </div>
        </div>
        
        <h2>Scenario Results</h2>
    """
    
    # Add scenario results
    for result in results:
        scenario_id = result.get("scenario_id", "Unknown")
        scenario_name = result.get("scenario_name", "Unknown Scenario")
        passed = result.get("passed", False)
        steps = result.get("steps", 0)
        timestamp = result.get("timestamp", "Unknown")
        
        header_class = "passed-header" if passed else "failed-header"
        status_text = "PASSED" if passed else "FAILED"
        
        html_content += f"""
        <div class="scenario">
            <div class="scenario-header {header_class}">
                <h3>Scenario {scenario_id}: {scenario_name}</h3>
                <span>{status_text}</span>
            </div>
            <div class="scenario-body">
                <p><strong>Steps:</strong> {steps}</p>
                <p class="timestamp">Timestamp: {timestamp}</p>
            </div>
        </div>
        """
    
    # Close HTML
    html_content += f"""
        <div class="footer">
            <p>Requify Document Processing Pipeline - Test Report</p>
        </div>
    </body>
    </html>
    """
    
    # Write the report file
    try:
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {report_file}", extra={"icon": "✅"})
        
        # Also create a link to the latest report
        latest_link = os.path.join(REPORTS_DIR, "latest_report.html")
        if os.path.exists(latest_link):
            os.remove(latest_link)
        
        # Create a copy of the report as latest_report.html
        with open(latest_link, 'w') as f:
            f.write(html_content)
        
        return True
    
    except Exception as e:
        logger.error(f"Error generating HTML report: {e}", extra={"icon": "❌"})
        return False

def main():
    """Main entry point for the script"""
    logger.info("Generating test report...", extra={"icon": "📊"})
    
    # Get latest test results
    results = get_latest_test_results()
    
    if not results:
        logger.warning("No test results found", extra={"icon": "⚠️"})
        return 1
    
    # Generate HTML report
    if generate_html_report(results):
        logger.info("Test report generated successfully", extra={"icon": "✅"})
        return 0
    else:
        logger.error("Failed to generate test report", extra={"icon": "❌"})
        return 1

if __name__ == "__main__":
    sys.exit(main()) 