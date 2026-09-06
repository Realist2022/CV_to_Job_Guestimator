"""Publish a LoRA adapter to Hugging Face with its provenance recorded in the repo.

Why this exists. The adapter Modal serves (modal_app/inference.py) has to be
identifiable months later, and the August 2026 exports were not: three HF
repos held three different training runs under names that did not say which
Ollama tag each one produced, and the GGUF export path overwrote a fixed
filename, so the source of `cv-guestimator:v2` no longer exists on disk. The
only thing that survived that mess intact was file hashes. So this script
refuses to upload without recording one, and writes the correspondence
between the adapter, its Ollama tag and its sha256 into the repo's own README
where the next person will actually find it.

This is for publishing a NEW build. The current default, v2, is already on
the Hub and is pinned by revision in modal_app/inference.py -- it needs no
re-upload, and re-uploading it would only add a fourth copy to disambiguate
later.

Dry run (default) -- validates, hashes, and prints the README it would write:
    uv run --with huggingface_hub python scripts/push_adapter_to_hf.py \
        --source C:/Projects/FT_MODEL2/models/checkpoints/<run-name> \
        --repo Realist2026/cv-guestimator-v4-lora \
        --ollama-tag cv-guestimator:v4

Add --yes to actually create the repo and upload. Needs HF_TOKEN in the
environment (already in .env for this project).

huggingface_hub is intentionally not a project dependency: nothing in src/
uploads to the Hub, and this runs by hand a few times a year. `uv run --with`
fetches it for the one invocation instead.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

# vLLM needs the adapter weights and the tokenizer the LoRA was trained
# against -- modal_app/inference.py passes ADAPTER_DIR to `--tokenizer` for
# exactly that reason, so an upload missing these serves the model under a
# different prompt rendering than it was trained on.
REQUIRED = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_readme(args, adapter_sha: str, rank: int, alpha: int, base: str) -> str:
    tag_line = (
        f"Serving this adapter should reproduce the Ollama tag `{args.ollama_tag}`."
        if args.ollama_tag
        else "No Ollama tag was recorded for this adapter."
    )
    return f"""---
library_name: peft
base_model: {base}
tags:
  - lora
  - cv-matching
---

# {args.repo.split("/")[-1]}

LoRA adapter for the CV-to-job guestimator. {tag_line}

## Provenance

| | |
| --- | --- |
| `adapter_model.safetensors` sha256 | `{adapter_sha}` |
| LoRA rank / alpha | {rank} / {alpha} |
| `base_model_name_or_path` | `{base}` |
| Uploaded from | `{args.source}` |
| Corresponding Ollama tag | `{args.ollama_tag or "unrecorded"}` |

The sha256 above is the identity of these weights. Check it before trusting
any claim about which training run this repo holds -- repo names and upload
dates have proven unreliable for this model.

## Serving

`base_model_name_or_path` names Unsloth's bnb-4bit base, which is a
*training-time* artifact. Serve the adapter against the unquantized mirror
`unsloth/Llama-3.2-3B-Instruct` instead; see `modal_app/inference.py` in the
CV_to_Job_Guestimator repo for the vLLM invocation. Note that the tokenizer
files uploaded here are *not* what that deployment serves -- it uses the
base's, because Unsloth writes these with a transformers newer than vLLM
0.11.0 can read. They are uploaded anyway so the adapter stays
self-describing; check they still match the base's before assuming that
substitution is free for a future build.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="local adapter directory")
    parser.add_argument("--repo", required=True, help="target repo id, e.g. user/name")
    parser.add_argument("--ollama-tag", help="the Ollama tag these weights correspond to")
    parser.add_argument(
        "--public",
        action="store_true",
        help="create the repo public (default private: it can always be opened up later, "
        "and Modal reads it with the HF_TOKEN already in its huggingface-secret)",
    )
    parser.add_argument("--yes", action="store_true", help="actually upload (default: dry run)")
    args = parser.parse_args()

    missing = [name for name in REQUIRED if not (args.source / name).is_file()]
    if missing:
        print(f"error: {args.source} is missing {', '.join(missing)}", file=sys.stderr)
        return 1

    import json

    config = json.loads((args.source / "adapter_config.json").read_text())
    adapter_sha = sha256(args.source / "adapter_model.safetensors")
    readme = build_readme(
        args, adapter_sha, config["r"], config["lora_alpha"], config["base_model_name_or_path"]
    )

    print(f"source:      {args.source}")
    print(f"repo:        {args.repo} ({'public' if args.public else 'private'})")
    print(f"adapter sha: {adapter_sha}")
    print(f"files:       {', '.join(REQUIRED)}")

    if not args.yes:
        print("\n--- README.md that would be written ---")
        print(readme)
        print("--- dry run: nothing uploaded. Re-run with --yes to publish. ---")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("error: HF_TOKEN is not set.", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    (args.source / "README.md").write_text(readme, encoding="utf-8")
    commit = api.upload_folder(
        repo_id=args.repo,
        folder_path=str(args.source),
        repo_type="model",
        # Training leftovers must not ride along: optimizer state and RNG
        # seeds are large and say nothing about how to serve the weights.
        ignore_patterns=["optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"],
        commit_message=f"Upload adapter for {args.ollama_tag or 'unrecorded tag'} ({adapter_sha[:12]})",
    )

    sha = getattr(commit, "oid", None) or str(commit)
    print(f"\nuploaded. commit: {sha}")
    print("Pin this in modal_app/inference.py:")
    print(f'    ADAPTER_REPO = "{args.repo}"')
    print(f'    ADAPTER_REVISION = "{sha}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
