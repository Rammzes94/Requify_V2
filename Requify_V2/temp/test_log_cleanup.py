#!/usr/bin/env python3
"""
test_log_cleanup.py

A simple test script to verify the regex patterns for log cleanup.
"""

import os
import sys
import re
import unittest

class LogCleanupTest(unittest.TestCase):
    """Test class for log cleanup patterns."""
    
    def get_sample_logs(self):
        """Return sample log lines for testing."""
        return """
logger.info(f"Generated {len(image_list)} images from PDF", extra={"icon": "✅"})
logger.info("--------------------------------------------------------------------------------", extra={"icon": "🔍"})
logger.info(f"Processing '{pdf_path}' to images...", extra={"icon": "🔄"})
logger.info(f"Converting PDF to images: {os.path.basename(pdf_path)}", extra={"icon": "📄"})
logger.info(f"Generating document title for {pdf_identifier} from page summaries...", extra={"icon": "🔤"})
logger.info(f"Generating document title for {pdf_identifier}...", extra={"icon": "🔤"})
print("Use pytorch device_name: mps")
logger.info("Connected to LanceDB", extra={"icon": "✅"})
logger.info(f"Not creating index: {table_name} has only {row_count} rows, minimum 256 required", extra={"icon": "⚠️"})
logger.info(f"Processing document: {document_id}", extra={"icon": "🚀"})
logger.info(f"Performing context-aware chunking with {len(chunks)} reference chunks", extra={"icon": "🧩"})
logger.info(f"🔄 Processing section {i+1}/{len(sections)}", extra={"icon": "🔄"})
"""

    def test_pdf_duplicate_images_pattern(self):
        """Test pattern for duplicate 'Generated images' messages."""
        pattern = r'logger\.info\(f"Generated \{len\([^)]+\)\} images from PDF", extra=\{"icon": "✅"\}\)'
        logs = self.get_sample_logs()
        matches = re.findall(pattern, logs)
        self.assertEqual(len(matches), 1, f"Expected 1 match for duplicate images pattern, got {len(matches)}")
    
    def test_pdf_separator_lines_pattern(self):
        """Test pattern for separator lines."""
        pattern = r'logger\.info\("[-]{80}", extra=\{"icon": "[^"]+"\}\)'
        logs = self.get_sample_logs()
        
        # Replace the actual separator with 80 dashes to match our pattern
        logs = logs.replace("--------------------------------------------------------------------------------", 
                           "-" * 80)
        
        matches = re.findall(pattern, logs)
        self.assertEqual(len(matches), 1, f"Expected 1 match for separator lines pattern, got {len(matches)}")
    
    def test_pdf_convert_images_patterns(self):
        """Test patterns for converting PDF to images."""
        patterns = [
            r'logger\.info\(f"Processing \'{[^}]+}\' to images\.\.\.", extra=\{"icon": "🔄"\}\)',
            r'logger\.info\(f"Converting PDF to images: \{[^}]+}\", extra=\{"icon": "📄"\}\)'
        ]
        logs = self.get_sample_logs()
        
        total_matches = 0
        for pattern in patterns:
            matches = re.findall(pattern, logs)
            total_matches += len(matches)
        
        self.assertEqual(total_matches, 2, f"Expected 2 matches for convert images patterns, got {total_matches}")
    
    def test_pdf_title_gen_patterns(self):
        """Test patterns for document title generation."""
        patterns = [
            r'logger\.info\(f"Generating document title for \{[^}]+\} from page summaries\.\.\.", extra=\{"icon": "🔤"\}\)',
            r'logger\.info\(f"Generating document title for \{[^}]+\}\.\.\.", extra=\{"icon": "🔤"\}\)'
        ]
        logs = self.get_sample_logs()
        
        total_matches = 0
        for pattern in patterns:
            matches = re.findall(pattern, logs)
            total_matches += len(matches)
        
        self.assertEqual(total_matches, 2, f"Expected 2 matches for title generation patterns, got {total_matches}")
    
    def test_replacement_format(self):
        """Test that replacement format is valid."""
        pattern = r'(logger\.info\(f"Generated \{len\([^)]+\)\} images from PDF", extra=\{"icon": "✅"\}\))'
        replacement = """# Avoid duplicate "Generated images" messages
_generated_images_logged = False
\\1
_generated_images_logged = True"""
        
        logs = self.get_sample_logs()
        new_logs = re.sub(pattern, replacement, logs)
        
        self.assertIn("_generated_images_logged = False", new_logs)
        self.assertIn("_generated_images_logged = True", new_logs)

if __name__ == "__main__":
    unittest.main() 