import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


METHODS = {
    "streamv2v_origin": "StreamV2V baseline",
    "gated_similarity_reverse": "Gated-StreamV2V reverse similarity gate",
    "gated_similarity_forward": "Gated similarity forward update gate",
    "confidence_gate": "Confidence gate",
}

PAPER_METRICS = {
    "streamv2v_origin": {"clip": 96.907, "warp": 110.677},
    "gated_similarity_reverse": {"clip": 96.911, "warp": 109.632},
    "gated_similarity_forward": {"clip": 96.224, "warp": 112.238},
    "confidence_gate": {"clip": 95.574, "warp": 114.059},
}

PAPER_HIGHRES = {
    ("streamv2v_origin", 1.0): {"memory_gb": 5.53, "time_s": 0.168},
    ("gated_similarity_reverse", 1.0): {"memory_gb": 4.83, "time_s": 0.156},
    ("streamv2v_origin", 2.0): {"memory_gb": 21.86, "time_s": 1.188},
    ("gated_similarity_reverse", 2.0): {"memory_gb": 8.43, "time_s": 1.028},
    ("streamv2v_origin", 3.0): {"memory_gb": None, "time_s": None, "status": "OOM"},
    ("gated_similarity_reverse", 3.0): {"memory_gb": 14.44, "time_s": 3.884},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize USTC paper-level reproduction.")
    parser.add_argument("--project_dir", default=".", help="Repository root.")
    parser.add_argument(
        "--report_dir",
        default="reports/ustc_paper_repro",
        help="Report directory relative to project_dir or absolute.",
    )
    parser.add_argument(
        "--output_root",
        default="vid2vid/output_ustc_paper",
        help="Generated video root relative to project_dir or absolute.",
    )
    parser.add_argument(
        "--eval_json",
        default="tools/user_study_upload/eval.json",
        help="Evaluation JSON array with vid_name entries.",
    )
    return parser.parse_args()


def resolve_path(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_eval_names(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item["vid_name"] for item in data if "vid_name" in item]


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def delta(observed: Optional[float], target: Optional[float]) -> Optional[float]:
    if observed is None or target is None:
        return None
    return observed - target


def collect_env() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        env["torch_error"] = str(exc)

    for module_name in ["diffusers", "transformers", "xformers", "cv2"]:
        try:
            module = __import__(module_name)
            env[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            env[f"{module_name}_error"] = str(exc)
    return env


def read_frame(video_path: Path, frame_index: int):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def make_labeled_image(frame, label: str):
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    label_h = 32
    canvas = Image.new("RGB", (image.width, image.height + label_h), "white")
    canvas.paste(image, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), label, fill="black")
    return canvas


def maybe_make_tennis_grid(project_dir: Path, report_dir: Path, output_root: Path) -> Optional[Path]:
    rows = [
        ("Input", project_dir / "vid2vid/source_video/tennis.mp4"),
        ("StreamV2V", output_root / "streamv2v_origin/tennis_ukiyoe_0.mp4"),
        ("Gated", output_root / "gated_similarity_reverse/tennis_ukiyoe_0.mp4"),
    ]
    if any(not path.exists() for _, path in rows):
        return None

    from PIL import Image

    frame_indices = [20, 24, 28, 32]
    cells = []
    for label, path in rows:
        row_cells = []
        for index in frame_indices:
            frame = read_frame(path, index)
            if frame is None:
                return None
            row_cells.append(make_labeled_image(frame, f"{label} f{index}"))
        cells.append(row_cells)

    cell_w = min(cell.width for row in cells for cell in row)
    cell_h = min(cell.height for row in cells for cell in row)
    resized = [[cell.resize((cell_w, cell_h)) for cell in row] for row in cells]
    grid = Image.new("RGB", (cell_w * len(frame_indices), cell_h * len(rows)), "white")
    for row_idx, row in enumerate(resized):
        for col_idx, cell in enumerate(row):
            grid.paste(cell, (col_idx * cell_w, row_idx * cell_h))

    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    out_path = figure_dir / "tennis_fig5_1_grid.jpg"
    grid.save(out_path, quality=95)
    return out_path


def metric_row(method: str, report_dir: Path) -> Dict[str, Any]:
    clip = load_json(report_dir / "metrics/clip" / f"{method}.clipscore")
    warp = load_json(report_dir / "metrics/warp" / f"{method}.warperror")
    return {
        "clip": clip.get("avg_score_x100") if clip else None,
        "clip_processed": clip.get("processed_videos") if clip else 0,
        "clip_missing": len(clip.get("missing", [])) if clip else None,
        "warp": warp.get("avg_error") if warp else None,
        "warp_processed": warp.get("processed_videos") if warp else 0,
        "warp_missing": len(warp.get("missing", [])) if warp else None,
    }


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    report_dir = resolve_path(project_dir, args.report_dir)
    output_root = resolve_path(project_dir, args.output_root)
    eval_json = resolve_path(project_dir, args.eval_json)
    report_dir.mkdir(parents=True, exist_ok=True)

    eval_names = load_eval_names(eval_json)
    env = collect_env()
    env_path = report_dir / "env_info.json"
    env_path.write_text(json.dumps(env, sort_keys=True, indent=4), encoding="utf-8")

    figure_path = maybe_make_tennis_grid(project_dir, report_dir, output_root)

    lines: List[str] = []
    lines.append("# USTC Gated-StreamV2V Paper Reproduction")
    lines.append("")
    lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    for key, value in env.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")

    lines.append("## Output Completion")
    lines.append("")
    lines.append("| Method | Expected | Present | Missing |")
    lines.append("| --- | ---: | ---: | ---: |")
    for method, label in METHODS.items():
        present = sum(
            1 for name in eval_names if (output_root / method / f"{name}.mp4").exists()
        )
        lines.append(f"| {label} | {len(eval_names)} | {present} | {len(eval_names) - present} |")
    lines.append("")

    lines.append("## Quantitative Metrics")
    lines.append("")
    lines.append("| Method | Paper CLIP | USTC CLIP | Delta | Paper Warp | USTC Warp | Delta |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method, label in METHODS.items():
        observed = metric_row(method, report_dir)
        target = PAPER_METRICS[method]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    fmt(target["clip"]),
                    fmt(observed["clip"]),
                    fmt(delta(observed["clip"], target["clip"])),
                    fmt(target["warp"]),
                    fmt(observed["warp"]),
                    fmt(delta(observed["warp"], target["warp"])),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## High-Resolution Benchmark")
    lines.append("")
    lines.append(
        "| Method | Scale | Paper Memory GB | USTC Peak Reserved GB | Paper s/frame | USTC s/frame | Status |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for method in ["streamv2v_origin", "gated_similarity_reverse"]:
        for scale in [1.0, 2.0, 3.0]:
            highres = load_json(report_dir / "highres" / f"{method}_scale{int(scale)}.json") or {}
            target = PAPER_HIGHRES[(method, scale)]
            lines.append(
                "| "
                + " | ".join(
                    [
                        METHODS[method],
                        fmt(scale, 1),
                        fmt(target.get("memory_gb")),
                        fmt(highres.get("peak_reserved_gb")),
                        fmt(target.get("time_s")),
                        fmt(highres.get("avg_time_seconds")),
                        highres.get("status", target.get("status", "-")),
                    ]
                )
                + " |"
            )
    lines.append("")

    lines.append("## Qualitative Check")
    lines.append("")
    if figure_path is None:
        lines.append("- Tennis frame grid is not available yet. Run the main methods first.")
    else:
        rel = figure_path.relative_to(report_dir)
        lines.append(f"- Tennis frame grid: [{rel}]({rel})")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- This report targets USTC reproducibility, not bitwise equality with the thesis machine.")
    lines.append("- Differences can come from GPU architecture, PyTorch/xFormers versions, public checkpoints, LoRA files, and video codecs.")
    lines.append("- CLIP values are reported as cosine similarity multiplied by 100 to match the thesis table scale.")
    lines.append("")

    summary_path = report_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", summary_path)
    print("Wrote", env_path)
    if figure_path:
        print("Wrote", figure_path)


if __name__ == "__main__":
    main()
