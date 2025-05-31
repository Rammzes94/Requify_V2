#!/bin/bash
# run_embedding_tests.sh
#
# This script automates running various document-level embedding test scenarios.
# It's designed to make testing easier without requiring user interaction.

# Display help
function show_help {
  echo "Document-Level Embedding Test Runner"
  echo ""
  echo "Usage: ./tests/embeddings/run_embedding_tests.sh [option]"
  echo ""
  echo "Options:"
  echo "  --list             List all available documents"
  echo "  --all              Run all tests with default document"
  echo "  --process DOC      Process a specific document"
  echo "  --similarity DOC1 [DOC2]  Test similarity between documents"
  echo "  --compare-all      Compare all document pairs for similarity"
  echo "  --help             Show this help message"
  echo ""
  echo "Examples:"
  echo "  ./tests/embeddings/run_embedding_tests.sh --all"
  echo "  ./tests/embeddings/run_embedding_tests.sh --process fighter_jet_rocket_launcher_spec_2.pdf"
  echo "  ./tests/embeddings/run_embedding_tests.sh --similarity fighter_jet_rocket_launcher_spec.pdf fighter_jet_rocket_launcher_spec_2.pdf"
  echo "  ./tests/embeddings/run_embedding_tests.sh --compare-all"
}

function run_test {
  echo "Running: $1"
  eval "$1"
  echo "----------------------------------------"
}

# Handle arguments
case "$1" in
  --list)
    run_test "python tests/embeddings/test_doc_embedding.py --list"
    ;;
    
  --all)
    # Run all tests with default document
    run_test "python tests/embeddings/test_doc_embedding.py --mode all --skip-hash-check"
    ;;
    
  --process)
    if [ -z "$2" ]; then
      echo "Error: Missing document name for --process"
      show_help
      exit 1
    fi
    run_test "python tests/embeddings/test_doc_embedding.py --mode process --doc $2 --skip-hash-check"
    ;;
    
  --similarity)
    if [ -z "$2" ]; then
      echo "Error: Missing document name for --similarity"
      show_help
      exit 1
    fi
    
    if [ -z "$3" ]; then
      # Compare with itself
      run_test "python tests/embeddings/test_doc_embedding.py --mode similarity --doc $2"
    else
      # Compare with specified second document
      run_test "python tests/embeddings/test_doc_embedding.py --mode similarity --doc $2 --doc2 $3"
    fi
    ;;
    
  --compare-all)
    # Get list of all PDF files
    echo "Retrieving list of PDF files..."
    pdf_files=$(find input/raw -name "*.pdf" -type f | sort | xargs -n1 basename)
    
    # Compare each pair of documents
    echo "Comparing all document pairs for similarity..."
    for doc1 in $pdf_files; do
      for doc2 in $pdf_files; do
        if [ "$doc1" != "$doc2" ]; then
          echo "Comparing $doc1 with $doc2"
          run_test "python tests/embeddings/test_doc_embedding.py --mode similarity --doc \"$doc1\" --doc2 \"$doc2\""
        fi
      done
    done
    ;;
    
  --help)
    show_help
    ;;
    
  *)
    echo "Unknown option: $1"
    show_help
    exit 1
    ;;
esac

exit 0 