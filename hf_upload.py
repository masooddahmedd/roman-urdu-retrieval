# Creates a PRIVATE Hugging Face dataset repo and uploads the query set and corpus. Needs HF_TOKEN
# (write access) in .env and your HF username as the first argument. Making it public is a manual
# step on the website, after the GitHub repo is public.
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
print("uploaded", dataset)
