#!/usr/bin/env python3
"""
test_results_reporter.py

This script generates enhanced HTML reports from E2E test results.
It processes test result JSON files and creates visualized reports with detailed information
about test outcomes, timings, and specific pass/fail criteria for each scenario.
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logging, get_logger, update_token_counters, get_token_usage, print_token_usage, reset_token_counters, setup_project_directory, generate_timestamp
setup_project_directory()

# Set up logging
logger = get_logger("Test_Reporter")

# Constants
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "test_results")
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "test_reports")

# Ensure report directory exists
os.makedirs(REPORT_DIR, exist_ok=True)

def load_test_results(result_file: str) -> Optional[Dict[str, Any]]:
    """
    Load test results from a JSON file
    
    Args:
        result_file: Path to the results JSON file
        
    Returns:
        Dictionary with the test results, or None if file not found/invalid
    """
    try:
        with open(result_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load test results: {str(e)}", extra={"icon": "❌"})
        return None

def generate_html_report(result_file: str) -> str:
    """
    Generate an HTML report from test results
    
    Args:
        result_file: Path to the JSON results file
        
    Returns:
        Path to the generated HTML report
    """
    # Load results
    results = load_test_results(result_file)
    if not results:
        logger.error(f"Failed to load results from {result_file}", extra={"icon": "❌"})
        return ""
    
    # Create output file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(REPORT_DIR, f"e2e_test_report_{timestamp}.html")
    
    # Calculate overall stats
    scenarios = results.get("results", [])
    total_scenarios = len(scenarios)
    passed_scenarios = sum(1 for s in scenarios if s.get('passed', False))
    success_rate = (passed_scenarios / total_scenarios) * 100 if total_scenarios > 0 else 0
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start building HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Requify E2E Test Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .summary {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border-left: 5px solid #5bc0de;
        }}
        .progress-bar {{
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .progress {{
            height: 100%;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-weight: bold;
        }}
        .scenario {{
            margin-bottom: 30px;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .passed {{
            border-left: 5px solid #5cb85c;
        }}
        .failed {{
            border-left: 5px solid #d9534f;
        }}
        .step {{
            margin: 10px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }}
        .step-passed {{
            border-left: 3px solid #5cb85c;
        }}
        .step-failed {{
            border-left: 3px solid #d9534f;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .check {{
            display: inline-block;
            width: 20px;
            height: 20px;
            text-align: center;
            border-radius: 50%;
            color: white;
            font-weight: bold;
        }}
        .check-passed {{
            background-color: #5cb85c;
        }}
        .check-failed {{
            background-color: #d9534f;
        }}
        .timestamp {{
            color: #777;
            font-size: 0.9em;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <h1>Requify E2E Test Results</h1>
    <div class="timestamp">Report generated on {timestamp}</div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Scenarios: {passed_scenarios}/{total_scenarios} passed ({success_rate:.1f}%)</p>
        <div class="progress-bar">
            <div class="progress" style="width: {success_rate}%; background-color: {'#5cb85c' if success_rate >= 80 else '#f0ad4e' if success_rate >= 50 else '#d9534f'}">
                {success_rate:.1f}%
            </div>
        </div>
    </div>
    
    <h2>Scenario Results</h2>
"""
    
    # Add each scenario
    for result in scenarios:
        scenario_id = result.get('scenario_id', 0)
        scenario_name = result.get('scenario_name', 'Unknown')
        passed = result.get('passed', False)
        steps = result.get('steps', [])
        
        html += f"""
    <div class="scenario {'passed' if passed else 'failed'}">
        <h3>Scenario {scenario_id}: {scenario_name} {'✅' if passed else '❌'}</h3>
"""
        
        # Add steps table
        html += """
        <table>
            <tr>
                <th>#</th>
                <th>File</th>
                <th>Result</th>
                <th>Duplicate</th>
                <th>New Chunks</th>
                <th>Duplicate Chunks</th>
                <th>Updated Chunks</th>
            </tr>
"""
        
        for i, step in enumerate(steps, 1):
            file_name = step.get('file', 'Unknown')
            step_passed = step.get('passed', False)
            is_duplicate = step.get('is_duplicate', False)
            new_chunks = step.get('new_chunks', False)
            duplicate_chunks = step.get('duplicate_chunks', False)
            updated_chunks = step.get('updated_chunks', False)
            
            html += f"""
            <tr>
                <td>{i}</td>
                <td>{file_name}</td>
                <td><span class="check {'check-passed' if step_passed else 'check-failed'}">{'✓' if step_passed else '✗'}</span></td>
                <td><span class="check {'check-passed' if is_duplicate else 'check-failed'}">{'✓' if is_duplicate else '✗'}</span></td>
                <td><span class="check {'check-passed' if new_chunks else 'check-failed'}">{'✓' if new_chunks else '✗'}</span></td>
                <td><span class="check {'check-passed' if duplicate_chunks else 'check-failed'}">{'✓' if duplicate_chunks else '✗'}</span></td>
                <td><span class="check {'check-passed' if updated_chunks else 'check-failed'}">{'✓' if updated_chunks else '✗'}</span></td>
            </tr>
"""
        
        html += """
        </table>
"""
        
        # If there are failures, display them
        failures = step.get('failures', [])
        if failures:
            html += """
        <h4>Failures:</h4>
        <ul>
"""
            for failure in failures:
                html += f"""
            <li>{failure}</li>
"""
            html += """
        </ul>
"""
        
        html += """
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    # Write the HTML to the output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(html)
    
    logger.info(f"HTML report generated at {output_file}", extra={"icon": "✅"})
    return output_file

def find_latest_results_file() -> Optional[str]:
    """
    Find the latest test results file
    
    Returns:
        Path to the latest results file, or None if no files found
    """
    result_files = glob.glob(os.path.join(RESULTS_DIR, "e2e_test_results_*.json"))
    if not result_files:
        return None
    
    # Sort by timestamp (newest first)
    result_files.sort(reverse=True)
    
    # Return the latest file
    return result_files[0]

def generate_consolidated_report():
    """
    Generate a consolidated HTML report for all test runs in the current session.
    This combines the results of all scenarios into a single report.
    """
    # Find all test result JSON files from the current session
    result_files = glob.glob(os.path.join(RESULTS_DIR, "e2e_test_results_*.json"))
    
    # Sort by timestamp (newest first)
    result_files.sort(reverse=True)
    
    if not result_files:
        logger.warning("No test result files found for consolidated report", extra={"icon": "⚠️"})
        return
    
    # Get the current timestamp for the report filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(RESULTS_DIR, f"consolidated_test_report_{timestamp}.html")
    
    # Load all result data
    all_results = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_results.append(data)
        except Exception as e:
            logger.error(f"Error loading result file {file_path}: {e}", extra={"icon": "❌"})
    
    # Group results by scenario ID
    scenarios = {}
    for result in all_results:
        scenario_id = result.get("scenario_id")
        if scenario_id not in scenarios or result.get("timestamp", "") > scenarios[scenario_id].get("timestamp", ""):
            scenarios[scenario_id] = result
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Requify Test Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }}
            .summary {{
                margin-bottom: 30px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 4px;
            }}
            .scenario-card {{
                margin-bottom: 20px;
                padding: 15px;
                border-radius: 4px;
                border-left: 5px solid #ddd;
                background-color: #fff;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .scenario-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }}
            .success {{
                border-left-color: #28a745;
            }}
            .failure {{
                border-left-color: #dc3545;
            }}
            .badge {{
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
                color: white;
            }}
            .badge-success {{
                background-color: #28a745;
            }}
            .badge-failure {{
                background-color: #dc3545;
            }}
            .step-list {{
                list-style-type: none;
                padding-left: 10px;
            }}
            .step-list li {{
                margin-bottom: 10px;
                padding-left: 20px;
                position: relative;
            }}
            .step-list li:before {{
                content: "";
                position: absolute;
                left: 0;
                top: 8px;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background-color: #6c757d;
            }}
            .step-success:before {{
                background-color: #28a745 !important;
            }}
            .step-failure:before {{
                background-color: #dc3545 !important;
            }}
            .timestamp {{
                color: #6c757d;
                font-size: 14px;
            }}
            .summary-metrics {{
                display: flex;
                gap: 20px;
                margin-top: 15px;
            }}
            .metric-box {{
                flex: 1;
                padding: 15px;
                border-radius: 4px;
                text-align: center;
            }}
            .metric-box h3 {{
                margin-top: 0;
                margin-bottom: 5px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
            }}
            .success-rate {{
                background-color: #e8f4f8;
            }}
            .passed-count {{
                background-color: #e8f8e8;
            }}
            .failed-count {{
                background-color: #f8e8e8;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Requify Document Processing Test Report</h1>
                <p class="timestamp">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="summary">
                <h2>Test Summary</h2>
                <p>Results from the latest run of each test scenario:</p>
                
                <div class="summary-metrics">
                    <div class="metric-box passed-count">
                        <h3>Passed</h3>
                        <div class="metric-value">{sum(1 for s in scenarios.values() if s.get("passed", False))}</div>
                    </div>
                    <div class="metric-box failed-count">
                        <h3>Failed</h3>
                        <div class="metric-value">{sum(1 for s in scenarios.values() if not s.get("passed", False))}</div>
                    </div>
                    <div class="metric-box success-rate">
                        <h3>Success Rate</h3>
                        <div class="metric-value">
                            {(sum(1 for s in scenarios.values() if s.get("passed", False)) / len(scenarios) * 100) if scenarios else 0:.0f}%
                        </div>
                    </div>
                </div>
            </div>
            
            <h2>Test Scenarios</h2>
    """
    
    # Add each scenario
    for scenario_id in sorted(scenarios.keys()):
        scenario = scenarios[scenario_id]
        status_class = "success" if scenario.get("passed", False) else "failure"
        badge_class = "badge-success" if scenario.get("passed", False) else "badge-failure"
        badge_text = "PASSED" if scenario.get("passed", False) else "FAILED"
        
        html_content += f"""
            <div class="scenario-card {status_class}">
                <div class="scenario-header">
                    <h3>Scenario {scenario_id}: {scenario.get("scenario_name", "Unknown")}</h3>
                    <span class="badge {badge_class}">{badge_text}</span>
                </div>
                <p>Executed on {scenario.get("timestamp", "Unknown").replace("_", " ")}</p>
                <p>Steps: {scenario.get("steps", 0)}</p>
            </div>
        """
    
    # Close the HTML
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write the report
    with open(report_file, 'w') as f:
        f.write(html_content)
        
    logger.info(f"Consolidated test report generated: {report_file}", extra={"icon": "✅"})
    return report_file

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate enhanced test reports")
    parser.add_argument("--results-dir", help="Directory containing test results", default=RESULTS_DIR)
    parser.add_argument("--file", help="Specific results file to process")
    parser.add_argument("--consolidated", action="store_true", help="Generate a consolidated report for all test runs")
    args = parser.parse_args()
    
    if args.results_dir:
        global RESULTS_DIR
        RESULTS_DIR = args.results_dir
    
    # Ensure the results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if args.file:
        # Process a specific file
        logger.info(f"Generating report for specific file: {args.file}", extra={"icon": "🔄"})
        if not os.path.exists(args.file):
            logger.error(f"Results file not found: {args.file}", extra={"icon": "❌"})
            return 1
        
        generate_html_report(args.file)
    elif args.consolidated:
        # Generate a consolidated report
        logger.info("Generating consolidated report for all test runs", extra={"icon": "🔄"})
        report_file = generate_consolidated_report()
        if report_file:
            logger.info(f"Consolidated report generated: {report_file}", extra={"icon": "✅"})
        else:
            logger.error("Failed to generate consolidated report", extra={"icon": "❌"})
            return 1
    else:
        # Process the latest results file
        logger.info("Finding latest results file...", extra={"icon": "🔄"})
        latest_file = find_latest_results_file()
        if latest_file:
            logger.info(f"Found latest results file: {latest_file}", extra={"icon": "✅"})
            generate_html_report(latest_file)
            # Also generate a consolidated report
            generate_consolidated_report()
        else:
            logger.error("No results files found", extra={"icon": "❌"})
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 