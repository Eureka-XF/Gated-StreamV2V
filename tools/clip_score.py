import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import cv2
from PIL import Image


DEFAULT_CLIP_MODEL = os.environ.get("GSV2V_CLIP_MODEL_ID", "openai/clip-vit-base-patch32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate CLIP frame consistency.")
    parser.add_argument(
        "--cuda_visible_devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override. Leave unset under Slurm.",
    )
    parser.add_argument("--device", default="cuda", help="Device to run the model on.")
    parser.add_argument(
        "--method_version",
        default="default",
        help="Method name used for the output JSON filename.",
    )
    parser.add_argument(
        "--set_file_path",
        default="user_study_upload/eval.json",
        help="Evaluation JSON array or JSONL path.",
    )
    parser.add_argument(
        "--edit_video_dir",
        default=None,
        help="Directory containing generated mp4 files. Defaults to ../vid2vid/output/<method_version>.",
    )
    parser.add_argument(
        "--clip_model_id_or_path",
        default=DEFAULT_CLIP_MODEL,
        help="CLIP model id or local path.",
    )
    parser.add_argument(
        "--output_log_dir",
        default="clip_score_log",
        help="Directory for the .clipscore JSON output.",
    )
    return parser.parse_args()


def load_eval_data(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
        if isinstance(loaded, list):
            return loaded
        raise ValueError(f"Expected a JSON array in {path}, got {type(loaded).__name__}")
    except json.JSONDecodeError:
        rows = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
        return rows


def create_vid_prompt_dict(json_data: List[Dict[str, Any]]) -> Dict[str, str]:
    video_prompts = {}
    for item in json_data:
        vid_name = item.get("vid_name")
        prompt = item.get("prompt")
        if vid_name and prompt:
            video_prompts[vid_name] = prompt
    return video_prompts


def read_video_frames(video_path: Path) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frames: List[Image.Image] = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = args.device
    method_version = args.method_version
    set_file_path = Path(args.set_file_path)
    edit_video_dir = (
        Path(args.edit_video_dir)
        if args.edit_video_dir is not None
        else Path("../vid2vid/output") / method_version
    )
    output_log_dir = Path(args.output_log_dir)
    output_log_dir.mkdir(parents=True, exist_ok=True)

    json_data = load_eval_data(set_file_path)
    video_maps = create_vid_prompt_dict(json_data)
    if not video_maps:
        raise SystemExit(f"No videos with vid_name/prompt found in {set_file_path}")

    print(f"method_version: {method_version}")
    print(f"set_file_path: {set_file_path}")
    print(f"edit_video_dir: {edit_video_dir}")
    print(f"clip_model_id_or_path: {args.clip_model_id_or_path}")

    model = CLIPModel.from_pretrained(args.clip_model_id_or_path).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model_id_or_path)
    model.eval()

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    consistency_scores: List[float] = []
    prompt_scores: List[float] = []
    missing: List[str] = []
    per_video: Dict[str, Any] = {}

    for vid_name, prompt in video_maps.items():
        video_path = edit_video_dir / f"{vid_name}.mp4"
        frames = read_video_frames(video_path)
        if len(frames) < 2:
            print(f"missing or too short: {video_path}")
            missing.append(vid_name)
            continue

        video_embs = []
        text_embeds = None
        for image in frames:
            with torch.no_grad():
                inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
                inputs = {key: value.to(device) for key, value in inputs.items()}
                outputs = model(**inputs)
            video_embs.append(outputs.image_embeds)
            text_embeds = outputs.text_embeds

        video_embs_tensor = torch.cat(video_embs, dim=0)
        prompt_score = cos(text_embeds, video_embs_tensor).mean().cpu().item()
        consistency_score = cos(video_embs_tensor[:-1], video_embs_tensor[1:]).mean().cpu().item()

        prompt_scores.append(prompt_score)
        consistency_scores.append(consistency_score)
        per_video[vid_name] = {
            "prompt": prompt,
            "clip_frame_consistency": consistency_score,
            "clip_prompt_score": prompt_score,
            "frames": len(frames),
        }
        print(vid_name, prompt, consistency_score)

    if not consistency_scores:
        raise SystemExit("No valid videos were scored.")

    avg_consistency = sum(consistency_scores) / len(consistency_scores)
    avg_prompt = sum(prompt_scores) / len(prompt_scores)
    out_json: Dict[str, Any] = {
        "method_version": method_version,
        "edit_video_dir": str(edit_video_dir),
        "set_file_path": str(set_file_path),
        "clip_model_id_or_path": args.clip_model_id_or_path,
        "processed_videos": len(consistency_scores),
        "expected_videos": len(video_maps),
        "missing": missing,
        "avg_score": avg_consistency,
        "avg_score_x100": avg_consistency * 100.0,
        "avg_prompt_score": avg_prompt,
        "videos": per_video,
    }

    output_path = output_log_dir / f"{method_version}.clipscore"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(out_json, file, sort_keys=True, indent=4)

    print("Number of videos", len(consistency_scores))
    print("Missing videos", len(missing))
    print("Avg consistency score", avg_consistency)
    print("Avg consistency score x100", avg_consistency * 100.0)
    print("Wrote", output_path)


if __name__ == "__main__":
    main()
