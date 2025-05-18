#!/bin/bash
# Script to run the processing script and generate artifacts

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create artifacts directory
mkdir -p artifacts

# Run the processing script
echo "Running MLPerf Optimizer processing script..."
python3 process.py

echo "Processing complete! Check the artifacts directory for results."