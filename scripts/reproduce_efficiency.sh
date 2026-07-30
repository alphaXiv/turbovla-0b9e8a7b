#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HOME="${TURBOVLA_RESULTS_DIR}/hf-cache"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

pod_index="${JOB_COMPLETION_INDEX:-0}"
expected_pods="${TURBOVLA_EXPECTED_PODS:-1}"
run_dir="${TURBOVLA_RESULTS_DIR}/efficiency"
model_dir="${TURBOVLA_RESULTS_DIR}/dinov3-vitb16"
model_spec="${TURBOVLA_RESULTS_DIR}/model_spec.json"

mkdir -p "${run_dir}" "${HF_HOME}"

echo "TURBOVLA_SETUP_BEGIN pod_index=${pod_index} expected_pods=${expected_pods}"
python -m pip install --quiet --no-cache-dir \
  "transformers==4.56.2" \
  "huggingface-hub>=0.34,<1" \
  "timm>=1.0" \
  "Pillow>=10" \
  "numpy>=1.26,<2" \
  "nvidia-ml-py>=12"

python - <<'PY'
import torch
import transformers

print(
    "TURBOVLA_ENV "
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"transformers={transformers.__version__} "
    f"cuda_available={torch.cuda.is_available()} gpu_count={torch.cuda.device_count()}"
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the efficiency reproduction")
PY

if [[ "${pod_index}" == "0" ]]; then
  rm -f "${model_spec}.tmp"
  MODEL_DIR="${model_dir}" MODEL_SPEC="${model_spec}" python - <<'PY'
import json
import os
from pathlib import Path

model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
model_dir = Path(os.environ["MODEL_DIR"])
spec_path = Path(os.environ["MODEL_SPEC"])
payload = {
    "model_id": model_id,
    "model_path": str(model_dir),
    "weights_source": "released_pretrained",
}
try:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_id,
        local_dir=model_dir,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.bin",
            "*.txt",
            "*.model",
        ],
    )
except Exception as exc:
    payload["weights_source"] = "shape_faithful_random_fallback"
    payload["download_error"] = f"{type(exc).__name__}: {exc}"
tmp = spec_path.with_suffix(".tmp")
tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
tmp.replace(spec_path)
print("TURBOVLA_MODEL_SPEC " + json.dumps(payload, sort_keys=True))
PY
else
  deadline=$((SECONDS + 1800))
  while [[ ! -s "${model_spec}" ]]; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for model setup from leader pod" >&2
      exit 1
    fi
    sleep 5
  done
fi

python scripts/benchmark_efficiency.py \
  --output-dir "${run_dir}" \
  --pod-index "${pod_index}" \
  --expected-pods "${expected_pods}" \
  --model-spec "${model_spec}"
