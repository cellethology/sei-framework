uv venv --python 3.9
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install selene selene_sdk docopt setuptools
uv add safetensors tqdm

# testing
# sh 1_sequence_prediction.sh <input-file> <genome> <output-dir> --cuda