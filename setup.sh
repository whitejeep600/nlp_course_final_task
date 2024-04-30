# Requires conda, poetry

conda env create -f environment.yaml
conda activate nlp_task
poetry install

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

mypy --install-types
