#!/bin/bash
set -e  # Exit on error

# Check if uv is installed, install if not
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    pip install uv
else
    echo "uv is already installed"
fi

# Create virtual environment and install dependencies
echo "Setting up virtual environment and installing dependencies..."
uv sync

# Note: Virtual environment activation won't persist after script exits
# Users should manually activate with: source .venv/bin/activate
echo ""
echo "Setup complete! To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To download the SEI model, run:"
echo "  bash download_data.sh"
