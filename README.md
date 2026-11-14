# Elephant-1 NF4

Code and documentation for [Agentware/elephant-1-nf4](https://huggingface.co/Agentware/elephant-1-nf4). Model weights and tokenizer assets are hosted on Hugging Face.

A text-only NF4 quantization of [baselabs/elephant-1](https://huggingface.co/baselabs/elephant-1) by Agentware, for structured decision tasks using the OneJev decision engine.

Base revision: `b3f8a93462fb1051a9de031e94a7ff380e254072`.

## Quantization

- Safetensors weight file: 10,005,707,616 bytes (about 9.32 GiB).
- 200 bitsandbytes Linear4bit modules, NF4 without nested/double quantization.
- BF16 compute and input embeddings; FP32 LM head.
- 48 protected `in_proj_a` / `in_proj_b` projections remain BF16; `conv1d` is excluded from quantization.
- `slot_lm` readout, block-bidirectional full-attention layers, causal linear-attention layers.
- Supplied adapter forces sequential branches and 16,384-token branch/request limits.

## Usage

Use a CUDA GPU with the supplied adapter and patched OneJev runtime. Export versions: PyTorch 2.11.0+cu130, Transformers 5.19.0, bitsandbytes 0.50.2, Accelerate 1.15.0 and Safetensors 0.8.0. Additional dependency pins are in `constraints.txt`.

Run setup in a dedicated Python/CUDA environment with a fresh runtime directory. `setup.sh` requires a fresh runtime directory and leaves existing checkouts untouched.

```bash
hf download Agentware/elephant-1-nf4 --local-dir elephant-1-nf4
mkdir runtime-work
bash elephant-1-nf4/setup.sh runtime-work
python -m pip install bitsandbytes==0.50.2
```

Setup installs OneJev at `81ce62f1597c91e46767d4d02d6ac2e18534fe94` with `serve.patch`, providing `qev.engine`.

```python
import sys
from pathlib import Path
model = Path("elephant-1-nf4").resolve()
sys.path.insert(0, str(model))
from nf4_adapter import load_quantized_engine
from qev.schema import SystemOneRequest

# Initialize the engine single-threaded (the loader patches class attributes).
engine = load_quantized_engine(model, device="cuda:0")
response, _ = engine.decide(SystemOneRequest(
    state="I was charged twice. Please refund the duplicate payment.",
    questions={"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
))
print(response.answers)
```

## Evaluation

Release checks cover file hashes, loader source and the serving patch. CUDA inference and benchmarks were not rerun for this release. The scores in `upstream/README.baselabs.md` are upstream BF16 results, not measurements of this NF4 checkpoint.

## Compatibility

This checkpoint uses the supplied decision-model loader, rather than a standard chat-generation interface. CPU, AMD GPUs, image/video input, vLLM, Ollama and llama.cpp have not been tested. The adapter caps branch/request length at 16,384 tokens; longer-context configurations are untested. Peak VRAM depends on sequence length and workload; weight-file size is not a VRAM requirement.

## Files and license

The artifact files and loader sources were SHA-256 verified. `artifact_manifest.json` and `nf4_audit.json` retain the original export metadata; `SHA256SUMS` lists release-file checksums.

Weight SHA-256: `5734f9a3e87f741e1d03f4a2309f79977f5a10c04696c2391f806aff7a1dda2b`.

The upstream model is MIT-licensed. Qwen and OneJev Apache-2.0 notices are retained under `upstream/`. See `NOTICE.md` for attribution.
