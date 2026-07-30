#!/usr/bin/env python3
"""Distributed synthetic-input efficiency reproduction for released TurboVLA.

Each Indexed Job pod owns one GPU. Index zero aggregates every worker result
from the shared volume into the terminal log consumed by ``orx logs``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock


PAPER = {
    "hardware": "NVIDIA RTX 4090",
    "batch_size": 1,
    "parameters_billion": 0.2,
    "latency_ms": 31.2,
    "inference_vram_gb": 0.9,
}
# BERT-base is used to build the released instruction cache. It is not resident
# in the online policy process, but the paper's 0.2B "total parameters" count
# includes the modality encoder while its sub-1GB deployment uses cached text.
BERT_BASE_PARAMETER_COUNT = 109_482_240
IMAGE_SIZE = 256
NUM_VIEWS = 2
TEXT_TOKENS = 32
TEXT_HIDDEN = 768
ACTION_DIM = 7
ACTION_HORIZON = 12
WARMUP_STEPS = 12
TIMED_STEPS = 1_000_000
PROGRESS_INTERVAL = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pod-index", type=int, required=True)
    parser.add_argument("--expected-pods", type=int, default=2)
    parser.add_argument("--model-spec", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--local-gpu", type=int)
    parser.add_argument("--global-index", type=int)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def process_gpu_memory_mib() -> float | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        current_pid = str(os.getpid())
        matches = []
        for line in output.splitlines():
            pieces = [piece.strip() for piece in line.split(",")]
            if len(pieces) >= 2 and pieces[0] == current_pid:
                matches.append(float(pieces[1]))
        return sum(matches) if matches else None
    except Exception:
        return None


def construct_model(model_spec: dict):
    import torch
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    from turbovla.models.turbovla import GroundingDINOVLA
    from turbovla.models import turbovla as turbovla_module

    common = dict(
        action_dim=ACTION_DIM,
        chunk_size=ACTION_HORIZON,
        text_hidden_dim=TEXT_HIDDEN,
        hidden_dim=256,
        nheads=8,
        dim_feedforward=2048,
        enhancer_inner_dim=1024,
        max_text_len=256,
        vla_feature_enhancer_layers=6,
        state_dim=8,
        num_state_tokens=2,
        text_dropout=0.0,
        fusion_dropout=0.0,
        fusion_droppath=0.1,
        freeze_vision_encoder=True,
        local_files_only=True,
    )
    if model_spec["weights_source"] == "released_pretrained":
        return GroundingDINOVLA(model_spec["model_path"], **common)

    # ViT-B/16 with four register tokens + CLS matches the released DINOv3
    # token geometry and parameterization. Random weights are sufficient for
    # latency, allocation, and parameter-count measurements.
    config = DINOv3ViTConfig(
        image_size=IMAGE_SIZE,
        patch_size=16,
        num_channels=3,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        num_register_tokens=4,
        use_mask_token=False,
    )
    backbone = DINOv3ViTModel(config)
    with mock.patch.object(
        turbovla_module.AutoModel,
        "from_pretrained",
        return_value=backbone,
    ):
        return GroundingDINOVLA("shape-faithful-random-dinov3-vitb16", **common)


def worker(args: argparse.Namespace) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.local_gpu)
    import numpy as np
    import torch

    torch.manual_seed(42 + int(args.global_index))
    np.random.seed(42 + int(args.global_index))
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"worker expected exactly one visible CUDA device, got {torch.cuda.device_count()}"
        )

    device = torch.device("cuda:0")
    model_spec = json.loads(args.model_spec.read_text(encoding="utf-8"))
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    model = construct_model(model_spec)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    paper_total_parameter_count = parameter_count + BERT_BASE_PARAMETER_COUNT
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    model.requires_grad_(False)

    generator = torch.Generator(device="cpu").manual_seed(20260730)
    cpu_images_u8 = torch.randint(
        0,
        256,
        (1, NUM_VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE),
        generator=generator,
        dtype=torch.uint8,
    )
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1)
    cpu_state = torch.zeros((1, 8), dtype=torch.float32)
    cpu_text = {
        "last_hidden_state": torch.randn(
            (1, TEXT_TOKENS, TEXT_HIDDEN), generator=generator, dtype=torch.float32
        ),
        "attention_mask": torch.ones((1, TEXT_TOKENS), dtype=torch.bool),
    }
    gpu_images = ((cpu_images_u8.float() / 255.0 - mean) / std).to(
        device=device, dtype=torch.bfloat16
    )
    gpu_state = cpu_state.to(device=device, dtype=torch.bfloat16)

    def forward_only():
        with torch.inference_mode():
            return model(cpu_text, {"dinov3": gpu_images}, gpu_state)

    for _ in range(WARMUP_STEPS):
        output = forward_only()
    torch.cuda.synchronize()

    reference = output.detach().float().cpu()
    repeated = forward_only()
    torch.cuda.synchronize()
    determinism_max_abs = float(
        (reference - repeated.detach().float().cpu()).abs().max().item()
    )

    cuda_times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    phase_started = time.perf_counter()
    for step in range(1, TIMED_STEPS + 1):
        start_event.record()
        output = forward_only()
        end_event.record()
        end_event.synchronize()
        cuda_times.append(float(start_event.elapsed_time(end_event)))
        if step % PROGRESS_INTERVAL == 0:
            print(
                "TURBOVLA_PROGRESS "
                + json.dumps(
                    {
                        "completed": step,
                        "elapsed_s": time.perf_counter() - phase_started,
                        "global_index": int(args.global_index),
                        "phase": "cuda_model",
                        "running_mean_ms": statistics.fmean(cuda_times),
                        "target": TIMED_STEPS,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    wall_times = []
    phase_started = time.perf_counter()
    for step in range(1, TIMED_STEPS + 1):
        torch.cuda.synchronize()
        started = time.perf_counter()
        pixels = ((cpu_images_u8.float() / 255.0 - mean) / std).to(
            device=device, dtype=torch.bfloat16
        )
        state = cpu_state.to(device=device, dtype=torch.bfloat16)
        with torch.inference_mode():
            policy_output = model(cpu_text, {"dinov3": pixels}, state)
        policy_output.float().cpu()
        torch.cuda.synchronize()
        wall_times.append((time.perf_counter() - started) * 1000.0)
        if step % PROGRESS_INTERVAL == 0:
            print(
                "TURBOVLA_PROGRESS "
                + json.dumps(
                    {
                        "completed": step,
                        "elapsed_s": time.perf_counter() - phase_started,
                        "global_index": int(args.global_index),
                        "phase": "end_to_end_policy",
                        "running_mean_ms": statistics.fmean(wall_times),
                        "target": TIMED_STEPS,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    allocated_peak = torch.cuda.max_memory_allocated() / (1024**3)
    reserved_peak = torch.cuda.max_memory_reserved() / (1024**3)
    allocated_steady = torch.cuda.memory_allocated() / (1024**3)
    reserved_steady = torch.cuda.memory_reserved() / (1024**3)
    process_mib = process_gpu_memory_mib()

    result = {
        "status": "ok",
        "global_index": int(args.global_index),
        "pod_index": int(args.pod_index),
        "local_gpu": int(args.local_gpu),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "weights_source": model_spec["weights_source"],
        "parameter_count": int(parameter_count),
        "online_policy_parameters_billion": parameter_count / 1e9,
        "cached_bert_parameters": BERT_BASE_PARAMETER_COUNT,
        "paper_total_parameter_count": int(paper_total_parameter_count),
        "paper_total_parameters_billion": paper_total_parameter_count / 1e9,
        "trainable_parameter_count_before_eval_freeze": int(trainable_parameter_count),
        "input_shape": [1, NUM_VIEWS, 3, IMAGE_SIZE, IMAGE_SIZE],
        "text_tokens": TEXT_TOKENS,
        "output_shape": list(output.shape),
        "output_finite": bool(torch.isfinite(output).all().item()),
        "determinism_max_abs": determinism_max_abs,
        "model_latency_ms_median": statistics.median(cuda_times),
        "model_latency_ms_mean": statistics.fmean(cuda_times),
        "model_latency_ms_p95": percentile(cuda_times, 0.95),
        "policy_wall_ms_median": statistics.median(wall_times),
        "policy_wall_ms_mean": statistics.fmean(wall_times),
        "policy_wall_ms_p95": percentile(wall_times, 0.95),
        "model_hz_from_median": 1000.0 / statistics.median(cuda_times),
        "cuda_peak_allocated_gb": allocated_peak,
        "cuda_peak_reserved_gb": reserved_peak,
        "cuda_steady_allocated_gb": allocated_steady,
        "cuda_steady_reserved_gb": reserved_steady,
        "nvidia_smi_process_gb": None if process_mib is None else process_mib / 1024.0,
        "paper_reference": PAPER,
    }
    output_path = args.output_dir / f"worker-{args.global_index:02d}.json"
    tmp_path = output_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(output_path)
    print("TURBOVLA_WORKER_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return 0


def launch_workers(args: argparse.Namespace) -> int:
    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu_count = torch.cuda.device_count()
    expected_gpus = int(os.environ.get("TURBOVLA_EXPECTED_GPUS_PER_POD", "8"))
    if gpu_count != expected_gpus:
        raise RuntimeError(
            f"pod expected {expected_gpus} visible GPUs, got {gpu_count}"
        )
    processes = []
    for local_gpu in range(gpu_count):
        global_index = args.pod_index * gpu_count + local_gpu
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--output-dir",
            str(args.output_dir),
            "--pod-index",
            str(args.pod_index),
            "--expected-pods",
            str(args.expected_pods),
            "--model-spec",
            str(args.model_spec),
            "--local-gpu",
            str(local_gpu),
            "--global-index",
            str(global_index),
        ]
        processes.append((global_index, subprocess.Popen(command)))

    failures = []
    for global_index, process in processes:
        code = process.wait()
        if code != 0:
            failures.append((global_index, code))
    if failures:
        raise RuntimeError(f"worker failures: {failures}")

    (args.output_dir / f"pod-{args.pod_index}.done").write_text(
        "ok\n", encoding="utf-8"
    )
    if args.pod_index != 0:
        print(f"TURBOVLA_POD_COMPLETE pod_index={args.pod_index}", flush=True)
        return 0

    expected_workers = args.expected_pods * gpu_count
    deadline = time.monotonic() + 2700
    while True:
        result_paths = sorted(args.output_dir.glob("worker-*.json"))
        done_paths = sorted(args.output_dir.glob("pod-*.done"))
        if len(result_paths) == expected_workers and len(done_paths) == args.expected_pods:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for {expected_workers} workers and "
                f"{args.expected_pods} pods; got {len(result_paths)} and {len(done_paths)}"
            )
        time.sleep(5)

    results = [
        json.loads(path.read_text(encoding="utf-8")) for path in result_paths
    ]
    print("TURBOVLA_ALL_RESULTS_BEGIN")
    for result in results:
        print(json.dumps(result, sort_keys=True))

    summary_fields = [
        "online_policy_parameters_billion",
        "paper_total_parameters_billion",
        "model_latency_ms_median",
        "policy_wall_ms_median",
        "model_hz_from_median",
        "cuda_peak_allocated_gb",
        "cuda_peak_reserved_gb",
    ]
    summary = {
        "status": "ok",
        "replicates": len(results),
        "paper_reference": PAPER,
        "hardware": sorted({result["gpu_name"] for result in results}),
        "weights_source": sorted({result["weights_source"] for result in results}),
        "output_shapes": sorted({tuple(result["output_shape"]) for result in results}),
    }
    for field in summary_fields:
        values = [float(result[field]) for result in results]
        summary[field] = {
            "min": min(values),
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }

    validations = {
        "all_workers_ok": all(result["status"] == "ok" for result in results),
        "replicate_count_expected": len(results) == expected_workers,
        "parameter_count_near_paper_0p2b": all(
            0.15 <= result["paper_total_parameters_billion"] <= 0.25
            for result in results
        ),
        "output_shape_1x12x7": all(
            result["output_shape"] == [1, 12, 7] for result in results
        ),
        "outputs_finite": all(result["output_finite"] for result in results),
        "deterministic_eval": all(
            result["determinism_max_abs"] == 0.0 for result in results
        ),
    }
    summary["validations"] = validations
    if not all(validations.values()):
        summary["status"] = "failed_validation"

    print("TURBOVLA_SUMMARY " + json.dumps(summary, sort_keys=True))
    print("TURBOVLA_ALL_RESULTS_END")
    return 0 if summary["status"] == "ok" else 2


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker(args)
    return launch_workers(args)


if __name__ == "__main__":
    raise SystemExit(main())
