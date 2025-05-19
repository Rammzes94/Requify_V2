# Requify Codebase Cleanup

This document provides instructions for cleaning up and reorganizing the Requify codebase to follow proper organization principles.

## Problem Statement

The current project structure has several issues:

1. Files are scattered across different directories without proper organization
2. There are duplicate files (same name and similar content in different locations)
3. Test files are mixed with source code in the `_02_src` directory
4. Utility tools are mixed with the main pipeline components
5. There are many test files that are not part of the e2e pipeline test

## Reorganization Principles

The reorganization follows these core principles:

1. **Source Directory (`_02_src/`)**: Contains only the core pipeline components
2. **Tools Directory (`tools/`)**: Contains all utility tools and scripts
3. **Tests Directory (`tests/`)**: Contains only necessary tests, primarily the e2e pipeline test
4. **De-duplication**: Eliminate duplicate files across directories

## How to Run the Cleanup

We've created a cleanup script that automates the reorganization process. Follow these steps:

1. First, make sure all your work is committed to version control (if using Git)
2. Run the cleanup script:

```bash
python tools/cleanup_codebase.py
```

3. The script will:
   - Back up all files before moving/deleting them to `backup/cleanup_backup/`
   - Move tool files from `_02_src/` to `tools/`
   - Move test files from `_02_src/` to `tests/`
   - Delete duplicate and unnecessary files

## Expected Outcome

After running the cleanup script, your codebase should have:

1. A clean `_02_src/` directory with only the core pipeline components
2. All utility tools organized in the `tools/` directory
3. Only necessary tests in the `tests/` directory
4. No duplicate files across directories

## Restoring Files (If Needed)

If something goes wrong or you need to restore a deleted file, all original files are backed up in the `backup/cleanup_backup/` directory. You can copy them back to their original locations if needed.

## Manual Verification

After running the cleanup script, you should manually:

1. Run the e2e pipeline test to ensure everything still works
2. Verify that no critical files were deleted
3. Update any import paths that might have broken during reorganization

## Questions or Issues

If you encounter any problems during or after the cleanup process, please:

1. Check the backup directory for your original files
2. Review the script output logs for errors
3. Contact the project maintainer for assistance 