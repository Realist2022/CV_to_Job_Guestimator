"""Modal deployment: the fine-tuned CV guestimator SLM behind an OpenAI-compatible API.

This is the cloud counterpart to `docker/ollama/Modelfile`. Both serve the same
LoRA; nothing in `src/` knows the difference, because `InstructorClient`
(src/services/llm_client.py) only ever speaks OpenAI-compatible HTTP. Pointing
the app here is a `configs/llm.yaml` entry, not a code change.

Deploy:
    uv run modal deploy modal_app/inference.py

Then put the printed URL (with `/v1` appended) and the API key into `.env`:
    MODAL_INFERENCE_URL=https://<workspace>--cv-guestimator-serve.modal.run/v1
    MODAL_API_KEY=<the same value as the Modal secret below>

Prerequisites -- two Modal secrets, created once:
    modal secret create huggingface-secret HF_TOKEN=hf_...
    modal secret create cv-guestimator-api-key VLLM_API_KEY=<a long random string>

NOTE ON WEIGHTS. `Realist2026/Llama-3.2-Semantic-Guestimator` is a *LoRA adapter*
(adapter_config.json + a 97MB adapter_model.safetensors), not a merged model. Its
`base_model_name_or_path` names Unsloth's bnb-4bit base, which is a training-time
artifact: serving a 4-bit bnb checkpoint under vLLM with LoRA on top is fragile and
slow. So BASE_MODEL below is the unquantized mirror of the same weights, and the
adapter is applied at serve time. This is the standard Unsloth deployment path.

`unsloth/Llama-3.2-3B-Instruct` is used rather than `meta-llama/Llama-3.2-3B-Instruct`
because the Meta repo is gated -- it 403s until the account behind HF_TOKEN has
accepted the license, which turns a first deploy into a support ticket. The Unsloth
mirror is the same weights, ungated.
"""

import subprocess

import modal

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER_REPO = "Realist2026/Llama-3.2-Semantic-Guestimator"

BASE_DIR = "/models/base"
ADAPTER_DIR = "/models/adapter"

# The name the app asks for. Deliberately NOT the same as SERVED_BASE_NAME: vLLM
# exposes the base model and each LoRA module as separate model ids on one
# endpoint, so `model: cv-guestimator` in configs/llm.yaml gets the fine-tune and
# `model: llama-3.2-3b-base` gets the stock model -- an A/B of the LoRA against
# its own base, on identical hardware, by changing one string in a task file.
LORA_NAME = "cv-guestimator"
SERVED_BASE_NAME = "llama-3.2-3b-base"

# Matches `PARAMETER num_ctx 8192` in docker/ollama/Modelfile, so a prompt that
# fits locally also fits here. Above the LoRA's 2048-token training window by
# design -- see that file's comment for the caveat, which carries over unchanged.
MAX_MODEL_LEN = 8192

# The adapter's rank from its adapter_config.json ("r": 16). vLLM preallocates
# LoRA slots against this, so it must be >= the real rank or loading fails.
MAX_LORA_RANK = 16

VLLM_PORT = 8000

hf_secret = modal.Secret.from_name("huggingface-secret")
api_key_secret = modal.Secret.from_name("cv-guestimator-api-key")


def _download_weights() -> None:
    """Bake base + adapter into the image at build time.

    Downloading at container start instead would put ~6.5GB of network transfer
    on the critical path of every cold start, which is exactly the latency that
    trips `max_execution_seconds` in configs/pipeline.yaml. Baked into an image
    layer, the weights are already on local disk when the container boots.
    """
    from huggingface_hub import snapshot_download

    snapshot_download(BASE_MODEL, local_dir=BASE_DIR)
    snapshot_download(ADAPTER_REPO, local_dir=ADAPTER_DIR)


vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # Pinned, not floating. vLLM moves fast and changes CLI flags and
        # numerics between minor versions; an unpinned image means a redeploy
        # months from now can quietly serve different output than the run
        # artifacts recorded against this endpoint. Same discipline as pinning
        # an Ollama tag instead of using :latest.
        "vllm==0.11.0",
        "huggingface_hub[hf_transfer]==0.35.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(_download_weights, secrets=[hf_secret])
)

app = modal.App("cv-guestimator")


@app.function(
    image=vllm_image,
    # A 3B in bf16 is ~6.4GB of weights; an L4's 24GB fits it with room for the
    # KV cache at 8192 context. Anything larger is paying for idle VRAM.
    gpu="L4",
    # How long a container lingers after its last request. The matching pipeline
    # makes three sequential LLM calls per run, and a human comparing candidates
    # makes several runs in a row, so a short window would cold-start mid-session.
    # Ten minutes keeps a working session warm; going to zero between sessions is
    # the point of serverless. Add `min_containers=1` here if you need
    # first-request latency and will pay for an always-on GPU.
    scaledown_window=10 * 60,
    timeout=20 * 60,
    secrets=[hf_secret, api_key_secret],
)
# One vLLM process batches concurrent requests far better than one-request-per-
# container does, and the harness can fan out runs. Inputs beyond this queue
# rather than cold-starting another GPU.
@modal.concurrent(max_inputs=8)
@modal.web_server(
    port=VLLM_PORT,
    # vLLM engine init + CUDA graph capture on a cold container, before it can
    # answer /health. Generous on purpose: exceeding it fails the deploy.
    startup_timeout=10 * 60,
)
def serve() -> None:
    import os

    subprocess.Popen(
        [
            "vllm",
            "serve",
            BASE_DIR,
            "--served-model-name",
            SERVED_BASE_NAME,
            # Serve the *adapter's* tokenizer, not the base's. The adapter repo
            # ships the tokenizer and chat_template.jinja the LoRA was actually
            # trained against (including its `<|finetune_right_pad_id|>` pad
            # token). Using the base's instead risks serving the model under a
            # marginally different prompt rendering than it was trained on --
            # the same class of train/serve skew that
            # scripts/build_training_dataset.py exists to avoid on the
            # prompt-construction side.
            "--tokenizer",
            ADAPTER_DIR,
            "--enable-lora",
            "--lora-modules",
            f"{LORA_NAME}={ADAPTER_DIR}",
            "--max-lora-rank",
            str(MAX_LORA_RANK),
            "--max-model-len",
            str(MAX_MODEL_LEN),
            # vLLM's own bearer-token auth, NOT Modal proxy auth. Modal's proxy
            # tokens are sent as `Modal-Key`/`Modal-Secret` headers, which the
            # OpenAI SDK inside InstructorClient will never send; vLLM's
            # --api-key checks `Authorization: Bearer`, which is exactly what
            # that SDK does send. So this drops straight into the existing
            # `api_key_env:` mechanism in configs/llm.yaml with no new code.
            "--api-key",
            os.environ["VLLM_API_KEY"],
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
        ]
    )
    # No wait/join: @modal.web_server polls the port and takes over once vLLM is
    # listening. Stop tokens are not passed here because vLLM reads the base
    # repo's generation_config.json, whose eos_token_id list already covers
    # <|end_of_text|>, <|eom_id|> and <|eot_id|> -- three of the four stops that
    # docker/ollama/Modelfile has to spell out by hand for Ollama.


@app.local_entrypoint()
def smoke_test() -> None:
    """Post-deploy check: `uv run modal run modal_app/inference.py`.

    Confirms the endpoint answers, that the LoRA id resolves (a typo'd
    --lora-modules name 404s here rather than silently serving the base model),
    and prints the cold-start cost so it can be compared against the
    `max_execution_seconds` budget in configs/pipeline.yaml before this model is
    promoted anywhere.
    """
    import json
    import os
    import time
    import urllib.request

    api_key = os.environ.get("MODAL_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set MODAL_API_KEY locally to the same value as the "
            "cv-guestimator-api-key Modal secret before running the smoke test."
        )

    url = serve.get_web_url().rstrip("/") + "/v1/chat/completions"
    payload = json.dumps(
        {
            "model": LORA_NAME,
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
            "temperature": 0.0,
            "max_tokens": 16,
        }
    ).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )

    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=15 * 60) as response:
        body = json.load(response)
    elapsed = time.monotonic() - started

    print(f"endpoint: {url}")
    print(f"model:    {body.get('model')}")
    print(f"reply:    {body['choices'][0]['message']['content']!r}")
    print(f"latency:  {elapsed:.1f}s (first call after a scaledown = cold start)")
