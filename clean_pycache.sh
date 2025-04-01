#!/bin/bash

# Find and remove all __pycache__ directories recursively from current directory
find . -type d -name "__pycache__" -exec rm -rf {} +

# Find and remove all .pyc files (compiled Python files)
find . -type f -name "*.pyc" -delete

echo "Cleaned all __pycache__ directories and .pyc files" 