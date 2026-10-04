# Creates PRIVATE Hugging Face repos and uploads the dataset and the demo Space. Needs HF_TOKEN
# (write access) in .env and your HF username as the first argument. Making them public is a
# manual step on the website, after the GitHub repo is public.
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
user = sys.argv[1]
api = HfApi(token=os.environ["HF_TOKEN"])
root = Path(__file__).parent

dataset = f"{user}/roman-urdu-retrieval-queries"
api.create_repo(dataset, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id=dataset, repo_type="dataset", folder_path=root / "hf_dataset" / "stage")

space = f"{user}/roman-urdu-retrieval"
api.create_repo(space, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
api.upload_folder(
    repo_id=space,
    repo_type="space",
    folder_path=root / "space",
    ignore_patterns=["build_space_data.py", "__pycache__"],
)
print("uploaded", dataset, space)
