import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torchvision.io import read_video
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
import torchvision.transforms as T
from tqdm import tqdm


DEFAULT_RAFT_WEIGHTS = os.environ.get(
    "GSV2V_RAFT_WEIGHTS",
    "../data/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth",
)


def coords_grid(b: int, h: int, w: int, device=None) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    grid = torch.stack([x, y], dim=0).float()
    grid = grid[None].repeat(b, 1, 1, 1)
    if device is not None:
        grid = grid.to(device)
    return grid


def bilinear_sample(
    img: torch.Tensor,
    sample_coords: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    return_mask: bool = False,
) -> torch.Tensor:
    if sample_coords.size(1) != 2:
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    _, _, h, w = sample_coords.shape
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1
    grid = torch.stack([x_grid, y_grid], dim=-1)

    sampled = torch.nn.functional.grid_sample(
        img,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)
        return sampled, mask
    return sampled


def flow_warp(
    feature: torch.Tensor,
    flow: torch.Tensor,
    mask: bool = False,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    _, _, h, w = feature.size()
    grid = coords_grid(feature.size(0), h, w, device=flow.device) + flow
    return bilinear_sample(feature, grid, padding_mode=padding_mode, return_mask=mask)


def forward_backward_consistency_check(
    fwd_flow: torch.Tensor,
    bwd_flow: torch.Tensor,
    alpha: float = 0.01,
    beta: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)
    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)
    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()
    bwd_occ = (diff_bwd > threshold).float()
    return fwd_occ, bwd_occ


def preprocess(batch: torch.Tensor) -> torch.Tensor:
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),
        ]
    )
    return transforms(batch)


def calculate_error(frame1: np.ndarray, frame2: np.ndarray, mask: torch.Tensor) -> float:
    mask_array = mask.numpy().astype(np.uint8)
    pixels_to_consider = mask_array == 0
    if not np.any(pixels_to_consider):
        return float("nan")
    return float(np.abs(frame1 - frame2)[pixels_to_consider].mean())


def calculate_warp_error_video(
    model: torch.nn.Module,
    ref_video_path: Path,
    edit_video_path: Path,
    device: str,
) -> float:
    ref_frames, _, _ = read_video(str(ref_video_path))
    edit_frames, _, _ = read_video(str(edit_video_path))
    if ref_frames.size(0) < 2 or edit_frames.size(0) < 2:
        raise ValueError(f"Video is too short: {edit_video_path}")

    ref_frames = ref_frames.permute(0, 3, 1, 2)
    edit_frames = edit_frames.permute(0, 3, 1, 2)
    ref_height, ref_width = ref_frames.shape[2], ref_frames.shape[3]
    edit_frames = torch.nn.functional.interpolate(
        edit_frames.float(),
        size=(ref_height, ref_width),
        mode="bilinear",
        align_corners=False,
    ).clamp(0, 255).to(torch.uint8)

    num_frames = min(edit_frames.shape[0], ref_frames.shape[0])
    ref_frames = ref_frames[:num_frames]
    edit_frames = edit_frames[:num_frames]

    errors: List[float] = []
    for i in range(num_frames - 1):
        fwd_batch = torch.stack([ref_frames[i], ref_frames[i + 1]])
        bwd_batch = torch.stack([ref_frames[i + 1], ref_frames[i]])
        fwd_batch = preprocess(fwd_batch).to(device)
        bwd_batch = preprocess(bwd_batch).to(device)

        with torch.no_grad():
            list_of_flows = model(fwd_batch, bwd_batch)
        predicted_flows = list_of_flows[-1]
        h, w = predicted_flows.shape[2:]
        fwd_occ, _ = forward_backward_consistency_check(
            predicted_flows[:1],
            predicted_flows[1:],
        )

        edit_image_1 = edit_frames[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        edit_image_2 = edit_frames[i + 1].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((grid_x, grid_y), axis=2)
        flow = predicted_flows[1].permute(1, 2, 0).cpu().detach().numpy()
        warped_grid = (grid + flow).astype(np.float32)
        warped_image = cv2.remap(edit_image_1, warped_grid, None, cv2.INTER_LINEAR)

        occlusion = fwd_occ[0].cpu().bool()
        warped_image[occlusion] = np.array([0, 0, 0], dtype=np.uint8)
        errors.append(calculate_error(warped_image, edit_image_2, occlusion))

    return float(np.nanmean(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate RAFT-based warp error.")
    parser.add_argument(
        "--cuda_visible_devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override. Leave unset under Slurm.",
    )
    parser.add_argument("--device", default="cuda", help="Device to use.")
    parser.add_argument(
        "--method_version",
        default="default",
        help="Method name used for the output JSON filename.",
    )
    parser.add_argument(
        "--set_file_path",
        default="user_study_upload/eval.json",
        help="Evaluation JSON array path.",
    )
    parser.add_argument(
        "--ref_video_dir",
        default="../vid2vid/source_video",
        help="Directory containing source/reference mp4 files.",
    )
    parser.add_argument(
        "--edit_video_dir",
        default=None,
        help="Directory containing generated mp4 files. Defaults to ../vid2vid/output/<method_version>.",
    )
    parser.add_argument(
        "--raft_weights",
        default=DEFAULT_RAFT_WEIGHTS,
        help="Local RAFT Large .pth file. If missing, torchvision C_T_SKHT_V2 is used.",
    )
    parser.add_argument(
        "--output_log_dir",
        default="warp_error_log",
        help="Directory for the .warperror JSON output.",
    )
    return parser.parse_args()


def load_eval_data(path: Path) -> List[Dict[str, Any]]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return loaded


def load_raft_model(weights_path: Path, device: str) -> torch.nn.Module:
    if weights_path.exists():
        print(f"Loading RAFT weights from {weights_path}")
        model = raft_large(weights=None, progress=False)
        state_dict = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"RAFT weights not found at {weights_path}; using torchvision C_T_SKHT_V2")
        model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=True)
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    device = args.device
    method_version = args.method_version
    set_file_path = Path(args.set_file_path)
    ref_video_dir = Path(args.ref_video_dir)
    edit_video_dir = (
        Path(args.edit_video_dir)
        if args.edit_video_dir is not None
        else Path("../vid2vid/output") / method_version
    )
    output_log_dir = Path(args.output_log_dir)
    output_log_dir.mkdir(parents=True, exist_ok=True)

    print(f"method_version: {method_version}")
    print(f"set_file_path: {set_file_path}")
    print(f"ref_video_dir: {ref_video_dir}")
    print(f"edit_video_dir: {edit_video_dir}")
    print(f"raft_weights: {args.raft_weights}")

    model = load_raft_model(Path(args.raft_weights), device)
    json_data = load_eval_data(set_file_path)

    video_error: List[float] = []
    missing: List[str] = []
    per_video: Dict[str, float] = {}
    for item in tqdm(json_data):
        src_vid_name = item["src_vid_name"]
        vid_name = item["vid_name"]
        ref_video_path = ref_video_dir / f"{src_vid_name}.mp4"
        edit_video_path = edit_video_dir / f"{vid_name}.mp4"

        if not ref_video_path.exists() or not edit_video_path.exists():
            print(f"missing: ref={ref_video_path.exists()} edit={edit_video_path.exists()} {vid_name}")
            missing.append(vid_name)
            continue

        cur_video_error = calculate_warp_error_video(
            model,
            ref_video_path,
            edit_video_path,
            device,
        )
        per_video[vid_name] = cur_video_error
        video_error.append(cur_video_error)
        print(vid_name, cur_video_error)

    if not video_error:
        raise SystemExit("No valid videos were scored.")

    avg_error = sum(video_error) / len(video_error)
    out_json: Dict[str, Any] = {
        "method_version": method_version,
        "ref_video_dir": str(ref_video_dir),
        "edit_video_dir": str(edit_video_dir),
        "set_file_path": str(set_file_path),
        "raft_weights": args.raft_weights,
        "processed_videos": len(video_error),
        "expected_videos": len(json_data),
        "missing": missing,
        "avg_error": avg_error,
        "videos": per_video,
    }

    output_path = output_log_dir / f"{method_version}.warperror"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(out_json, file, sort_keys=True, indent=4)

    print(f"Avg warp error of {method_version} is {avg_error}")
    print("Missing videos", len(missing))
    print("Wrote", output_path)


if __name__ == "__main__":
    main()
