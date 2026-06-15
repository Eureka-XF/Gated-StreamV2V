import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_MODEL_ID = os.environ.get("GSV2V_MODEL_ID", "runwayml/stable-diffusion-v1-5")
DEFAULT_LCM_LORA_ID = os.environ.get(
    "GSV2V_LCM_LORA_ID",
    "latent-consistency/lcm-lora-sdv1-5",
)


def str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple Gated-StreamV2V video edits from a JSONL file."
    )
    parser.add_argument("--json_file", required=True, help="Path to the JSONL task file.")
    parser.add_argument(
        "--output_dir",
        default="./output/default",
        help="Directory to save generated videos.",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override. Leave unset under Slurm.",
    )
    parser.add_argument(
        "--model_id",
        default=DEFAULT_MODEL_ID,
        help="Base diffusion model id or path. Overrides per-row model_id.",
    )
    parser.add_argument(
        "--lcm_lora_id",
        default=DEFAULT_LCM_LORA_ID,
        help="LCM-LoRA id or path passed to main.py. Use empty string to disable.",
    )
    parser.add_argument(
        "--cache_interval",
        type=int,
        default=1,
        help="Feature-bank update interval.",
    )
    parser.add_argument(
        "--noise_strength",
        default=None,
        help="Override noise_strength for all rows.",
    )
    parser.add_argument(
        "--diffusion_steps",
        default=None,
        help="Override diffusion_steps for all rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Random seed used by main.py.",
    )
    parser.add_argument(
        "--use_cached_attn",
        type=str_to_bool,
        default=True,
        help="Whether to use cached attention.",
    )
    parser.add_argument(
        "--cached_attn_style",
        choices=["origin", "confidence", "similarity"],
        default="similarity",
        help="Cached attention implementation.",
    )
    parser.add_argument(
        "--reverse_tag",
        type=str_to_bool,
        default=True,
        help="Similarity gate direction. True is the thesis default.",
    )
    parser.add_argument(
        "--use_attn_concat",
        type=str_to_bool,
        default=True,
        help="Whether to concatenate self-attention features.",
    )
    parser.add_argument(
        "--use_feature_injection",
        type=str_to_bool,
        default=True,
        help="Whether to use feature injection.",
    )
    parser.add_argument(
        "--feature_similarity_threshold",
        type=float,
        default=0.98,
        help="Feature similarity threshold.",
    )
    parser.add_argument(
        "--feature_injection_strength",
        type=float,
        default=0.5,
        help="Feature injection strength for the original StreamV2V path.",
    )
    parser.add_argument(
        "--ttt_lr",
        type=float,
        default=1.0,
        help="Scale factor for beta in gated attention.",
    )
    parser.add_argument(
        "--acceleration",
        choices=["none", "xformers", "tensorrt"],
        default="xformers",
        help="Acceleration backend.",
    )
    parser.add_argument(
        "--skip_existing",
        type=str_to_bool,
        default=True,
        help="Skip rows whose output mp4 already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run at most this many rows after filtering. 0 means all rows.",
    )
    parser.add_argument(
        "--only_vid_name",
        default=None,
        help="Run a single task by vid_name, useful for smoke tests.",
    )
    parser.add_argument(
        "--python_executable",
        default=sys.executable,
        help="Python executable used to launch main.py.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry count for each video task after a subprocess failure.",
    )
    parser.add_argument(
        "--retry_delay",
        type=float,
        default=10.0,
        help="Seconds to wait between task retries.",
    )
    parser.add_argument(
        "--dry_run",
        type=str_to_bool,
        default=False,
        help="Print commands without running them.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    return rows


def bool_arg(value: bool) -> str:
    return "True" if value else "False"


def filtered_rows(rows: Iterable[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected = list(rows)
    if args.only_vid_name:
        selected = [row for row in selected if row.get("vid_name") == args.only_vid_name]
        if not selected:
            raise ValueError(f"vid_name not found in {args.json_file}: {args.only_vid_name}")
    if args.limit and args.limit > 0:
        selected = selected[: args.limit]
    return selected


def row_value(row: Dict[str, Any], key: str, override: Any) -> str:
    value = row.get(key) if override is None else override
    if value is None:
        raise ValueError(f"Missing required field: {key}")
    return str(value)


def main() -> None:
    args = parse_arguments()
    script_dir = Path(__file__).resolve().parent
    json_file = Path(args.json_file)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = filtered_rows(load_jsonl(json_file), args)
    print(f"Loaded {len(rows)} task(s) from {json_file}", flush=True)
    print(f"Output directory: {output_dir.resolve()}", flush=True)
    print(f"Subprocess working directory: {script_dir}", flush=True)
    print(f"cached_attn_style: {args.cached_attn_style}", flush=True)
    print(f"reverse_tag: {bool_arg(args.reverse_tag)}", flush=True)

    completed = 0
    skipped = 0
    for row in rows:
        vid_name = row.get("vid_name")
        if not vid_name:
            raise ValueError(f"Task row has no vid_name; use ori_batch_eval.py style data: {row}")

        output_video = output_dir / f"{vid_name}.mp4"
        if args.skip_existing and output_video.exists() and output_video.stat().st_size > 0:
            print(f"video already exists, skip: {output_video}", flush=True)
            skipped += 1
            continue

        file_path = row_value(row, "file_path", None)
        src_vid_name = row_value(row, "src_vid_name", None)
        input_video = str(Path(file_path) / f"{src_vid_name}.mp4")
        model_id = args.model_id or row.get("model_id") or DEFAULT_MODEL_ID

        command = [
            args.python_executable,
            str(script_dir / "main.py"),
            "--input",
            input_video,
            "--prompt",
            row_value(row, "prompt", None),
            "--video_name",
            str(vid_name),
            "--output_dir",
            str(output_dir),
            "--model_id",
            str(model_id),
            "--diffusion_steps",
            row_value(row, "diffusion_steps", args.diffusion_steps),
            "--noise_strength",
            row_value(row, "noise_strength", args.noise_strength),
            "--acceleration",
            args.acceleration,
            "--use_cached_attn",
            bool_arg(args.use_cached_attn),
            "--cache_maxframes",
            "1",
            "--use_tome_cache",
            "True",
            "--do_add_noise",
            "True",
            "--guidance_scale",
            "1.0",
            "--cache_interval",
            str(args.cache_interval),
            "--use_attn_concat",
            bool_arg(args.use_attn_concat),
            "--use_feature_injection",
            bool_arg(args.use_feature_injection),
            "--feature_similarity_threshold",
            str(args.feature_similarity_threshold),
            "--feature_injection_strength",
            str(args.feature_injection_strength),
            "--ttt_lr",
            str(args.ttt_lr),
            "--cached_attn_style",
            args.cached_attn_style,
            "--reverse_tag",
            bool_arg(args.reverse_tag),
            "--seed",
            str(args.seed),
        ]
        if args.cuda_visible_devices is not None:
            command.extend(["--cuda_visible_devices", str(args.cuda_visible_devices)])
        if args.lcm_lora_id:
            command.extend(["--lcm_lora_id", args.lcm_lora_id])

        print("Running:", " ".join(command), flush=True)
        if args.dry_run:
            continue
        for attempt in range(args.retries + 1):
            try:
                subprocess.run(command, check=True, cwd=script_dir)
                break
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                if attempt >= args.retries:
                    raise
                wait_seconds = max(args.retry_delay, 0.0)
                print(
                    f"Task failed for {vid_name} on attempt {attempt + 1}/{args.retries + 1}: {exc}. "
                    f"Retrying in {wait_seconds:g}s...",
                    flush=True,
                )
                time.sleep(wait_seconds)
        completed += 1

    print(f"Done. completed={completed} skipped={skipped} total={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
