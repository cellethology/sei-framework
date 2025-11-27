#!/bin/sh

# Ensure aria2c is installed
if ! command -v aria2c >/dev/null 2>&1; then
  echo "aria2c not found. Installing aria2c..."
  sudo apt-get update && sudo apt-get install -y aria2
fi

# Trained Sei model
aria2c -x 16 -s 16 -k 1M \
  -o sei_model.tar.gz \
  "https://zenodo.org/record/4906997/files/sei_model.tar.gz"

tar -xzvf sei_model.tar.gz
