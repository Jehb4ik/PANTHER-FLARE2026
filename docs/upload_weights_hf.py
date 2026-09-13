"""Upload the trained weights to a HuggingFace model repository.

Usage: python upload_weights_hf.py --repo <user>/PANTHER-FLARE2026 --model-dir ../docker/model
Requires `huggingface-cli login` (write token) beforehand.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--model-dir", type=Path, default=Path("../docker/model"))
ap.add_argument("--card", type=Path, default=Path("hf_model_card.md"))
args = ap.parse_args()

api = HfApi()
api.create_repo(args.repo, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=args.repo, folder_path=str(args.model_dir), path_in_repo=".",
                  allow_patterns=["plans.json", "dataset.json", "fold_*/checkpoint_final.pth"])
api.upload_file(repo_id=args.repo, path_or_fileobj=str(args.card), path_in_repo="README.md")
print(f"uploaded to https://huggingface.co/{args.repo}")
