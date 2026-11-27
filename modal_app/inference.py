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

ADAPTER_REPO/ADAPTER_REVISION below already name the weights behind
`cv-guestimator:v2`; read the comment there before changing either. To publish
a future build as its own clearly-named repo, scripts/push_adapter_to_hf.py
uploads an adapter and records its sha256 in the repo's README, then prints the
commit sha to pin here.

NOTE ON WEIGHTS. ADAPTER_REPO holds a *LoRA adapter* (adapter_config.json + a
97MB adapter_model.safetensors), not a merged model -- which is also why it
cannot be produced from the GGUF that Ollama serves: that is a Q4_K_M
quantization of the *merged* weights, one lossy step further down the chain,
and there is no path back from it to an adapter. Its
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
from pathlib import Path

import modal

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"

# The adapter this endpoint serves, and the exact commit of it.
#
# THE REVISION IS THE IDENTITY, NOT THE REPO NAME. The head of this repo is a
# later training run -- the build promoted to `cv-guestimator:v3` and then
# discarded. The pinned commit below is the one that corresponds to
# `cv-guestimator:v2`, the promoted default in configs/llm.yaml.
#
# How that was established, since no name or date says it:
#   - Both FT_MODEL2 export scripts read a single `training.output_dir` that
#     every run overwrites, so an adapter's identity survives only in what was
#     pushed before the next run clobbered it (FT_MODEL2 commit 3d0deaf, "give
#     each training run a name that identifies it", is the fix for that).
#   - This commit pushed that directory at 2026-08-26T23:23Z; the GGUF behind
#     `cv-guestimator:v2` (Ollama blob sha256-5166523f...) was written 76
#     minutes later, with no training run in between -- the next one started
#     ~4h afterwards and produced the v3 adapter now at this repo's head.
#   - Its adapter_model.safetensors (sha256 a230b14d...) exists in no other
#     repo and nowhere on disk: the local checkpoints directory was overwritten
#     by that later run. This commit is the only surviving copy of v2.
#
# So: do not repoint this at a branch. A branch head here has already changed
# out from under the tag it was supposed to mean, once.
ADAPTER_REPO = "Realist2026/cv-guestimator-llama3.2-lora"
ADAPTER_REVISION: str | None = "4170cda7cc8aa15cca66fded6340ab16757f4109"

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
    # Imported here, not at module top: huggingface_hub is installed into the
    # Modal image by vllm_image's .pip_install below, and this function only
    # ever executes there (via .run_function at image build time). The local
    # venv has no reason to carry it, so a top-level import would break the
    # `modal deploy` that has to import this file locally. The pyright ignore
    # says the same thing to Pylance, which resolves imports against .venv.
    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    snapshot_download(BASE_MODEL, local_dir=BASE_DIR)
    snapshot_download(ADAPTER_REPO, local_dir=ADAPTER_DIR, revision=ADAPTER_REVISION)


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
    # Last, so editing the middleware does not rebuild the 6.5GB weights layer.
    .add_local_file(Path(__file__).parent / "vllm_auth.py", "/root/vllm_auth.py")
)

app = modal.App("cv-guestimator")

MINUTES = 60


def _auth_headers() -> dict[str, str]:
    import os

    return {"Authorization": f"Bearer {os.environ['VLLM_API_KEY']}"}


def _wait_ready(proc: subprocess.Popen) -> None:
    """Block until vLLM answers /health, or raise if the process died."""
    import time
    import urllib.request

    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {proc.returncode} during start-up")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=2):
                return
        except Exception:
            time.sleep(1)


def _post(path: str, payload: dict | None = None, timeout: float = 300) -> None:
    import json
    import urllib.request

    data = json.dumps(payload).encode() if payload is not None else b""
    request = urllib.request.Request(
        f"http://127.0.0.1:{VLLM_PORT}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", **_auth_headers()},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


@app.cls(
    image=vllm_image,
    # A 3B in bf16 is ~6.4GB of weights; an L4's 24GB fits it with room for the
    # KV cache at 8192 context. Anything larger is paying for idle VRAM.
    #
    # A10G is the fallback, tried only when no L4 is free. On 2026-11-26 a
    # request sat "scheduling" for 21 minutes waiting for an L4, so every
    # visitor in that window hit a timeout however well the warm-up worked.
    # The A10G is also 24GB and runs bf16, so the same flags fit unchanged; it
    # costs a little more per hour, paid only when the L4 pool is empty.
    gpu=["L4", "A10G"],
    # How long a container lingers after its last request. Three minutes spans
    # the pause between runs in a working session; an idle GPU bills exactly
    # like a busy one (~$0.76/hr for an L4), so longer is mostly waste.
    # `min_containers=1` removes cold starts entirely and pays for a GPU 24/7
    # (~$18/day). Only worth it for an endpoint serving real users.
    scaledown_window=3 * MINUTES,
    timeout=20 * MINUTES,
    secrets=[hf_secret, api_key_secret],
    # GPU MEMORY SNAPSHOTS. The slow part of a cold start was never the
    # weights (they are baked into the image): it was vLLM's engine start,
    # LoRA load and, without --enforce-eager, torch.compile plus CUDA graph
    # capture. With snapshots, Modal runs `start()` below once per deploy,
    # saves the container's CPU and GPU memory, and restores that image on
    # every later cold start instead of booting vLLM again.
    #
    # The FIRST cold start after each `modal deploy` is slower than before,
    # because it does the full boot and then takes the snapshot. Every one
    # after that restores. GPU snapshots are an alpha Modal feature; if a
    # restore ever fails, delete these two lines to go back to a plain boot.
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
# One vLLM process batches concurrent requests far better than one-request-per-
# container does, and the harness can fan out runs. Inputs beyond this queue
# rather than cold-starting another GPU.
@modal.concurrent(max_inputs=8)
class VllmServer:
    @modal.enter(snap=True)
    def start(self) -> None:
        """Boot vLLM, warm every code path, then put it to sleep for the snapshot."""
        import os

        env = {
            **os.environ,
            # Registers /sleep and /wake_up, which the snapshot flow needs.
            # Dev mode also exposes other admin routes, which is why the
            # RequireApiKey middleware below guards every path, not just /v1.
            "VLLM_SERVER_DEV_MODE": "1",
            # Modal's guidance for snapshotting code that uses torch.compile.
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            # So `--middleware vllm_auth.RequireApiKey` can import the module
            # copied to /root/vllm_auth.py by the image definition above.
            "PYTHONPATH": "/root",
        }
        self.vllm_proc = subprocess.Popen(
            [
                "vllm",
                "serve",
                BASE_DIR,
                "--served-model-name",
                SERVED_BASE_NAME,
                # No --tokenizer: the base's tokenizer is used, and for these
                # two repos that is the same tokenizer, not a compromise.
                #
                # The adapter's own copy cannot be loaded here at all. Unsloth
                # wrote it with a transformers/tokenizers far newer than vLLM
                # 0.11.0 pins: its tokenizer_config.json names a
                # `TokenizersBackend` class the installed library has never
                # heard of, and its tokenizer.json is a format the bundled
                # `tokenizers` cannot read. Both failures land during engine
                # init, so the container never boots.
                #
                # Serving the base's copy costs nothing here, which was checked
                # rather than assumed: the adapter's chat_template.jinja is
                # byte-identical to `unsloth/Llama-3.2-3B-Instruct`'s once CRLF
                # is normalised to LF, and both declare the same pad and eos
                # tokens. Re-check that if ADAPTER_REPO ever changes.
                "--enable-lora",
                "--lora-modules",
                f"{LORA_NAME}={ADAPTER_DIR}",
                "--max-lora-rank",
                str(MAX_LORA_RANK),
                "--max-model-len",
                str(MAX_MODEL_LEN),
                # No --enforce-eager any more. Eager mode skipped torch.compile
                # and CUDA graph capture (~90s) because every cold start paid
                # for them again. With a memory snapshot that work happens once
                # per deploy and is restored from the snapshot, so generation
                # gets the faster compiled path for free.
                #
                # Structured outputs: the app asks for a JSON schema per call
                # (configs/llm.yaml `structured_output: json_schema`), and
                # xgrammar constrains decoding to it.
                #
                # Whitespace is deliberately left free. Forbidding it
                # (`disable_any_whitespace`) was tried on 2026-11-27 and broke
                # the output: the LoRA learned JSON written with spaces after
                # `:` and `,`, so with those spaces banned it lost track of
                # where a string ended and wrote the next keys INTO the
                # string, e.g. target_job_title =
                # 'AI engineer","target_overall_years":null,'. It also
                # extracted far fewer requirements. Keep decoding close to
                # the format the model was trained on.
                "--structured-outputs-config",
                '{"backend": "xgrammar"}',
                # Required for /sleep and /wake_up.
                "--enable-sleep-mode",
                # Puts /sleep, /wake_up and the other dev-mode routes behind
                # the same bearer key as /v1. See modal_app/vllm_auth.py.
                "--middleware",
                "vllm_auth.RequireApiKey",
                # vLLM's own bearer-token auth for /v1, NOT Modal proxy auth.
                # Modal's proxy tokens are sent as `Modal-Key`/`Modal-Secret`
                # headers, which the OpenAI SDK inside InstructorClient never
                # sends; vLLM's --api-key checks `Authorization: Bearer`, which
                # is exactly what that SDK does send.
                "--api-key",
                os.environ["VLLM_API_KEY"],
                "--host",
                "0.0.0.0",
                "--port",
                str(VLLM_PORT),
            ],
            env=env,
        )
        _wait_ready(self.vllm_proc)

        # Exercise the LoRA and the JSON-schema path a few times so the
        # compiled kernels, CUDA graphs and grammar compiler are all warm in
        # the snapshot, not built on a visitor's first request.
        for _ in range(3):
            _post(
                "/v1/chat/completions",
                {
                    "model": LORA_NAME,
                    "messages": [{"role": "user", "content": "Reply with JSON."}],
                    "max_tokens": 16,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "warmup",
                            "schema": {
                                "type": "object",
                                "properties": {"ok": {"type": "boolean"}},
                                "required": ["ok"],
                            },
                        },
                    },
                },
            )

        # Level 1 moves the weights to CPU memory and drops the KV cache, so
        # the snapshot is small and the cache is rebuilt fresh on restore.
        _post("/sleep?level=1")

    @modal.enter(snap=False)
    def wake(self) -> None:
        """Runs after every restore: bring the weights back onto the GPU."""
        _post("/wake_up")
        _wait_ready(self.vllm_proc)

    # `label` keeps the public URL exactly what it was when `serve` was a plain
    # function (https://<workspace>--cv-guestimator-serve.modal.run), so the
    # MODAL_INFERENCE_URL secret on Cloud Run does not need to change.
    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES, label="cv-guestimator-serve")
    def serve(self) -> None:
        # No body: vLLM is already listening, started in `start()` above.
        # Stop tokens are not passed because vLLM reads the base repo's
        # generation_config.json, whose eos_token_id list already covers
        # <|end_of_text|>, <|eom_id|> and <|eot_id|>.
        pass

    @modal.exit()
    def stop(self) -> None:
        self.vllm_proc.terminate()


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

    url = VllmServer().serve.get_web_url().rstrip("/") + "/v1/chat/completions"
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
