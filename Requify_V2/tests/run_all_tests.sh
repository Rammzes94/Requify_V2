#!/bin/bash
# run_all_tests.sh
#
# This script runs all test scenarios for the Requify document processing pipeline
# and generates a summary of the results.

echo "===== REQUIFY DOCUMENT PROCESSING TEST SUITE ====="
echo "Starting test run at $(date)"
echo

# Clean the database before running tests
echo "Cleaning database..."
python src/_00_lancedb_admin/reset_lancedb.py

# Keep track of passed/failed scenarios
passed=0
failed=0
total=6  # Total number of scenarios

# Run each scenario
for i in $(seq 1 $total); do
  echo
  echo "===== Running Scenario $i ====="
  if python tests/e2e/test_scenarios.py --scenario $i; then
    echo "✅ Scenario $i PASSED"
    ((passed++))
  else
    echo "❌ Scenario $i FAILED"
    ((failed++))
  fi
done

# Print summary
echo
echo "===== TEST RESULTS SUMMARY ====="
echo "Total scenarios: $total"
echo "Passed: $passed"
echo "Failed: $failed"
echo
echo "Success rate: $((passed * 100 / total))%"
echo

# Generate HTML report if available
if [ -f "tools/test_utils/results_reporter.py" ]; then
  echo "Generating HTML report..."
  python tools/test_utils/results_reporter.py
  echo "Report generated in output/test_results/"
fi

echo "Test run completed at $(date)" 