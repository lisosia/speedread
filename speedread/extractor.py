from __future__ import annotations

import base64
import difflib
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .models import PageResult
from .prompts import get_prompt


@dataclass
class ExtractParams:
    analysis_fps: int = 10
    analysis_long_side: int = 640
    rotation_degrees: int = 0
    min_interval_s: float = 0.1
    max_interval_s: float = 3.0
    llm_base_url: str = "http://127.0.0.1:1234"
    llm_model: str = "qwen/qwen3-vl-8b"
    llm_timeout_s: int = 60
    llm_prompt_key: str = "general"
    or_window: int = 6
    min_peak_distance_s: float = 0.12
    peak_mad_k: float = 3.0
    boundary_motion_weight: float = 0.5
    boundary_change_weight: float = 0.5
    search_start_frames: int = 1
    search_end_s: float = 0.4
    output_per_peak: int = 1
    use_quad_in_score: bool = False
    enable_warp: bool = True
    use_center_crop_fallback: bool = False
    motion_weight: float = 0.7
    quad_weight: float = 0.15


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


def _safe_fps(raw_fps: float) -> float:
    if raw_fps <= 0 or raw_fps > 240:
        return 30.0
    return raw_fps


def _resize_long_side(frame: np.ndarray, long_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= long_side:
        return frame
    if h >= w:
        new_h = long_side
        new_w = int(w * (long_side / h))
    else:
        new_w = long_side
        new_h = int(h * (long_side / w))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def rotate_frame(frame: np.ndarray, rotation_degrees: int) -> np.ndarray:
    rotation = rotation_degrees % 360
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def decode_analysis_frames(
    video_path: str,
    analysis_fps: int,
    long_side: int,
    rotation_degrees: int = 0,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tuple[List[np.ndarray], List[Dict[str, int]], VideoInfo]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    fps = _safe_fps(float(cap.get(cv2.CAP_PROP_FPS)))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_info = VideoInfo(width=width, height=height, fps=fps, frame_count=frame_count)

    step = max(1, int(round(fps / float(analysis_fps))))
    frames: List[np.ndarray] = []
    infos: List[Dict[str, int]] = []

    idx = 0
    while True:
        if should_stop and should_stop():
            break
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            time_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if time_ms <= 0:
                time_ms = int((idx / fps) * 1000)
            frame_rotated = rotate_frame(frame, rotation_degrees)
            frame_small = _resize_long_side(frame_rotated, long_side)
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            infos.append({"frame_index": idx, "time_ms": time_ms})
        idx += 1
        if progress_cb and frame_count > 0 and idx % 50 == 0:
            percent = int(min(15, (idx / frame_count) * 15))
            progress_cb(percent, "Decoding analysis frames")

    cap.release()
    return frames, infos, video_info


def _smooth_series(series: List[float], window: int = 5) -> List[float]:
    if window <= 1 or len(series) < 2:
        return series[:]
    half = window // 2
    padded = [series[0]] * half + series + [series[-1]] * half
    smoothed: List[float] = []
    for i in range(len(series)):
        window_vals = padded[i : i + window]
        smoothed.append(float(sum(window_vals)) / float(len(window_vals)))
    return smoothed


def _auto_threshold(series: List[float], mad_k: float) -> float:
    arr = np.array(series, dtype=np.float32)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad <= 1e-6:
        mad = 1.0
    return med + mad_k * mad


def compute_motion_series(
    frames: List[np.ndarray],
    or_window: int,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tuple[List[int], List[float]]:
    if len(frames) < 2:
        return [0] * len(frames), [0.0] * len(frames)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_diffs: List[np.ndarray] = []

    for t in range(1, len(frames)):
        if should_stop and should_stop():
            break
        diff = cv2.absdiff(frames[t], frames[t - 1])
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=1)
        _, diff_bin = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_diffs.append(diff_bin)
        if progress_cb and t % 50 == 0:
            progress_cb(25, "Analyzing motion")

    B: List[int] = []
    window = deque(maxlen=or_window)
    for t, diff_bin in enumerate(bin_diffs):
        if should_stop and should_stop():
            break
        window.append(diff_bin)
        merged = window[0].copy()
        for item in list(window)[1:]:
            merged = cv2.bitwise_or(merged, item)
        B.append(int(cv2.countNonZero(merged)))
        if progress_cb and t % 50 == 0:
            progress_cb(30, "Accumulating motion")

    B_raw = [0] + B
    B_smooth = _smooth_series([float(v) for v in B_raw], window=5)
    return B_raw, B_smooth


def pick_peaks(
    series: List[float],
    min_distance_frames: int,
    mad_k: float,
) -> List[int]:
    if len(series) < 3:
        return []

    threshold = _auto_threshold(series, mad_k)
    candidates: List[int] = []
    for i in range(1, len(series) - 1):
        if series[i] >= series[i - 1] and series[i] > series[i + 1] and series[i] >= threshold:
            candidates.append(i)

    if not candidates:
        return []

    peaks: List[int] = []
    for idx in candidates:
        if not peaks:
            peaks.append(idx)
            continue
        if idx - peaks[-1] >= min_distance_frames:
            peaks.append(idx)
        else:
            if series[idx] > series[peaks[-1]]:
                peaks[-1] = idx
    return peaks


def _laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-6:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _bit_count(value: int) -> int:
    if hasattr(int, "bit_count"):
        return value.bit_count()
    return bin(value).count("1")


def _dhash(gray: np.ndarray, hash_size: int = 8) -> int:
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def compute_change_series(
    frames: List[np.ndarray],
    hash_size: int = 8,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tuple[List[int], List[float]]:
    if not frames:
        return [], []

    hashes: List[int] = []
    for idx, frame in enumerate(frames):
        if should_stop and should_stop():
            break
        hashes.append(_dhash(frame, hash_size))
        if progress_cb and idx % 50 == 0:
            progress_cb(28, "Analyzing content change")

    change_raw = [0]
    for i in range(1, len(hashes)):
        change_raw.append(_bit_count(hashes[i] ^ hashes[i - 1]))

    change_smooth = _smooth_series([float(v) for v in change_raw], window=5)
    return change_raw, change_smooth


def _llm_request(base_url: str, payload: Dict[str, object], timeout_s: int) -> Dict[str, object]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    return text


def _text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def transcribe_frame(frame: np.ndarray, params: ExtractParams) -> str:
    if frame.ndim == 2:
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        bgr = frame
    ok, buffer = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame for LLM")
    image_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")

    prompt = get_prompt(params.llm_prompt_key)
    payload = {
        "model": params.llm_model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ],
    }

    response = _llm_request(params.llm_base_url, payload, params.llm_timeout_s)
    content = ""
    try:
        message = response["choices"][0]["message"]["content"]
        if isinstance(message, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in message
            )
        else:
            content = str(message)
    except Exception as exc:
        raise RuntimeError("Unexpected LLM response format") from exc
    return content.strip()


def transcribe_image_file(image_path: str, params: ExtractParams) -> str:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return transcribe_frame(image, params)


def compute_text_change_series(
    frames: List[np.ndarray],
    params: ExtractParams,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tuple[List[str], List[float], List[float]]:
    if not frames:
        return [], [], []

    texts_raw: List[str] = []
    texts_norm: List[str] = []
    total = len(frames)

    for idx, frame in enumerate(frames):
        if should_stop and should_stop():
            break
        text = transcribe_frame(frame, params)
        texts_raw.append(text.strip())
        texts_norm.append(_normalize_text(text))
        if progress_cb and idx % 20 == 0:
            progress_cb(35, f"LLM {idx + 1}/{total}")

    change_raw = [0.0]
    for i in range(1, len(texts_norm)):
        sim = _text_similarity(texts_norm[i], texts_norm[i - 1])
        change_raw.append(1.0 - sim)

    change_smooth = _smooth_series(change_raw, window=5)
    return texts_raw, change_raw, change_smooth


def segment_by_text_change(
    times_ms: List[int],
    change_series: List[float],
    min_interval_s: float,
    max_interval_s: float,
    mad_k: float,
) -> List[int]:
    if not times_ms or not change_series:
        return []

    min_ms = max(0.0, min_interval_s) * 1000.0
    max_ms = max(min_ms, max_interval_s) * 1000.0
    threshold = _auto_threshold(change_series, mad_k)

    boundaries = [0]
    last_idx = 0
    candidate_idx: Optional[int] = None
    candidate_val = -1.0

    for i in range(1, len(change_series)):
        elapsed = times_ms[i] - times_ms[last_idx]
        value = change_series[i]

        if elapsed >= min_ms and value > candidate_val:
            candidate_val = value
            candidate_idx = i

        if elapsed >= min_ms and value >= threshold:
            if i != last_idx:
                boundaries.append(i)
                last_idx = i
            candidate_idx = None
            candidate_val = -1.0
            continue

        if elapsed >= max_ms:
            if candidate_idx is None:
                candidate_idx = i
            if candidate_idx != last_idx:
                boundaries.append(candidate_idx)
                last_idx = candidate_idx
            candidate_idx = None
            candidate_val = -1.0

    return boundaries


def _quick_quad_score(gray: np.ndarray) -> float:
    quad = detect_page_quad(gray)
    return 1.0 if quad is not None else 0.0


def select_frames(
    frames: List[np.ndarray],
    motion_series: List[float],
    peaks: List[int],
    params: ExtractParams,
) -> List[Dict[str, object]]:
    selections: List[Dict[str, object]] = []
    if not peaks:
        return selections

    search_end_frames = max(1, int(round(params.search_end_s * params.analysis_fps)))

    for peak in peaks:
        start = min(peak + params.search_start_frames, len(frames) - 1)
        end = min(peak + search_end_frames, len(frames) - 1)
        if start > end:
            continue

        sharp_vals: List[float] = []
        motion_vals: List[float] = []
        quad_vals: List[float] = []
        indices = list(range(start, end + 1))

        for idx in indices:
            sharp = _laplacian_variance(frames[idx])
            motion = float(motion_series[idx])
            quad_score = _quick_quad_score(frames[idx]) if params.use_quad_in_score else 0.0
            sharp_vals.append(sharp)
            motion_vals.append(motion)
            quad_vals.append(quad_score)

        sharp_norm = _normalize(sharp_vals)
        motion_norm = _normalize(motion_vals)

        scores: List[Tuple[int, float]] = []
        for i, idx in enumerate(indices):
            score = sharp_norm[i] - params.motion_weight * motion_norm[i] + params.quad_weight * quad_vals[i]
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        for idx, _score in scores[: params.output_per_peak]:
            i = idx - start
            selections.append(
                {
                    "analysis_index": idx,
                    "pce_peak_index": peak,
                    "score_components": {
                        "sharpness": float(sharp_vals[i]),
                        "motion": float(motion_vals[i]),
                        "quad_score": float(quad_vals[i]),
                    },
                }
            )
    return selections


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_page_quad(image: np.ndarray) -> Optional[np.ndarray]:
    if image is None:
        return None
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    img_h, img_w = gray.shape[:2]
    min_area = img_w * img_h * 0.08
    best = None
    best_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        if area > best_area:
            best_area = area
            best = approx

    if best is None:
        return None

    pts = best.reshape(4, 2).astype("float32")
    ordered = order_points(pts)

    width_a = np.linalg.norm(ordered[2] - ordered[3])
    width_b = np.linalg.norm(ordered[1] - ordered[0])
    height_a = np.linalg.norm(ordered[1] - ordered[2])
    height_b = np.linalg.norm(ordered[0] - ordered[3])

    width = max(width_a, width_b)
    height = max(height_a, height_b)
    if width < img_w * 0.2 or height < img_h * 0.2:
        return None

    ratio = width / max(height, 1.0)
    if ratio < 0.4 or ratio > 2.8:
        return None

    return ordered


def warp_perspective(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    width_a = np.linalg.norm(quad[2] - quad[3])
    width_b = np.linalg.norm(quad[1] - quad[0])
    height_a = np.linalg.norm(quad[1] - quad[2])
    height_b = np.linalg.norm(quad[0] - quad[3])

    max_width = int(max(width_a, width_b))
    max_height = int(max(height_a, height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def center_crop(image: np.ndarray, scale: float = 0.9) -> np.ndarray:
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    x0 = max(0, (w - new_w) // 2)
    y0 = max(0, (h - new_h) // 2)
    return image[y0 : y0 + new_h, x0 : x0 + new_w].copy()


def extract_highres_frame(
    cap: cv2.VideoCapture,
    frame_index: Optional[int] = None,
    time_ms: Optional[int] = None,
    rotation_degrees: int = 0,
) -> np.ndarray:
    if frame_index is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    elif time_ms is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)

    ok, frame = cap.read()
    if not ok and time_ms is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        ok, frame = cap.read()

    if not ok:
        raise RuntimeError("Failed to decode high-res frame")
    return rotate_frame(frame, rotation_degrees)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def extract_pages(
    video_path: str,
    params: ExtractParams,
    work_dir: Optional[str] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    on_item: Optional[Callable[[PageResult], None]] = None,
) -> Tuple[str, List[PageResult]]:
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="speedread_")

    ensure_dir(work_dir)
    pages_dir = os.path.join(work_dir, "pages_tmp")
    ensure_dir(pages_dir)

    if progress_cb:
        progress_cb(5, "Decoding analysis frames")
    frames, infos, video_info = decode_analysis_frames(
        video_path,
        params.analysis_fps,
        params.analysis_long_side,
        rotation_degrees=params.rotation_degrees,
        progress_cb=progress_cb,
        should_stop=should_stop,
    )

    if should_stop and should_stop():
        return work_dir, []

    if progress_cb:
        progress_cb(20, "Running local LLM transcription")
    texts, _change_raw, change_smooth = compute_text_change_series(
        frames,
        params,
        progress_cb=progress_cb,
        should_stop=should_stop,
    )

    if should_stop and should_stop():
        return work_dir, []

    fps = max(1, params.analysis_fps)
    times_ms = [int(i * (1000.0 / fps)) for i in range(len(frames))]
    boundaries = segment_by_text_change(
        times_ms,
        change_smooth,
        params.min_interval_s,
        params.max_interval_s,
        params.peak_mad_k,
    )

    if progress_cb:
        progress_cb(50, f"Selecting frames ({len(boundaries)})")

    selections: List[Dict[str, object]] = []
    if not boundaries:
        if progress_cb:
            progress_cb(95, "No text segments found")
        return work_dir, []

    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(frames)
        if end <= start:
            continue
        mid = start + (end - start) // 2
        score_val = change_smooth[mid] if mid < len(change_smooth) else 0.0
        selections.append(
            {
                "analysis_index": mid,
                "pce_peak_index": start,
                "score_components": {"text_change": float(score_val)},
            }
        )

    if not selections:
        if progress_cb:
            progress_cb(95, "No frames selected")
        return work_dir, []

    if progress_cb:
        progress_cb(60, "Extracting high-res frames")

    results: List[PageResult] = []
    cap = cv2.VideoCapture(video_path)
    for idx, sel in enumerate(selections, start=1):
        if should_stop and should_stop():
            break
        info = infos[sel["analysis_index"]]
        frame = extract_highres_frame(
            cap,
            frame_index=info["frame_index"],
            time_ms=info["time_ms"],
            rotation_degrees=params.rotation_degrees,
        )

        quad_points = None
        warp_applied = False
        if params.enable_warp:
            quad = detect_page_quad(frame)
            if quad is not None:
                frame = warp_perspective(frame, quad)
                quad_points = quad.astype(int).tolist()
                warp_applied = True

        if not warp_applied and params.use_center_crop_fallback:
            frame = center_crop(frame)

        image_path = os.path.join(pages_dir, f"page_{idx:04d}.png")
        cv2.imwrite(image_path, frame)

        text_value = None
        if sel["analysis_index"] < len(texts):
            text_value = texts[sel["analysis_index"]]
        result = PageResult(
            image_path=image_path,
            timestamp_ms=int(info["time_ms"]),
            pce_peak_index=int(sel["pce_peak_index"]),
            score_components=sel["score_components"],
            warp_applied=warp_applied,
            quad_points=quad_points,
            ocr_text=text_value,
        )
        results.append(result)
        if on_item:
            on_item(result)

        if progress_cb:
            percent = 60 + int((idx / max(1, len(selections))) * 30)
            progress_cb(percent, f"Extracted {idx}/{len(selections)}")

    cap.release()
    if progress_cb:
        progress_cb(95, "Finalizing")

    return work_dir, results


def extract_single_frame(
    video_path: str,
    time_ms: int,
    params: ExtractParams,
    work_dir: str,
    index: int,
) -> PageResult:
    ensure_dir(work_dir)
    pages_dir = os.path.join(work_dir, "pages_tmp")
    ensure_dir(pages_dir)

    cap = cv2.VideoCapture(video_path)
    frame = extract_highres_frame(cap, time_ms=time_ms, rotation_degrees=params.rotation_degrees)
    cap.release()

    quad_points = None
    warp_applied = False
    if params.enable_warp:
        quad = detect_page_quad(frame)
        if quad is not None:
            frame = warp_perspective(frame, quad)
            quad_points = quad.astype(int).tolist()
            warp_applied = True

    if not warp_applied and params.use_center_crop_fallback:
        frame = center_crop(frame)

    image_path = os.path.join(pages_dir, f"page_{index:04d}.png")
    cv2.imwrite(image_path, frame)

    return PageResult(
        image_path=image_path,
        timestamp_ms=int(time_ms),
        pce_peak_index=-1,
        score_components={"sharpness": 0.0, "motion": 0.0, "quad_score": 0.0},
        warp_applied=warp_applied,
        quad_points=quad_points,
        ocr_text=None,
    )


def export_results(
    results: Iterable[PageResult],
    output_dir: str,
    source_video_path: str,
    export_pdf: bool = False,
) -> Tuple[bool, str]:
    pages_dir = os.path.join(output_dir, "pages")
    ensure_dir(pages_dir)

    output_images: List[str] = []

    for idx, result in enumerate(results, start=1):
        dst_image = os.path.join(pages_dir, f"page_{idx:04d}.png")
        with open(result.image_path, "rb") as src_f:
            with open(dst_image, "wb") as dst_f:
                dst_f.write(src_f.read())
        output_images.append(dst_image)

        meta = result.to_json()
        meta["page_index"] = idx
        meta["source_video_path"] = source_video_path
        dst_json = os.path.join(pages_dir, f"page_{idx:04d}.json")
        with open(dst_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)

    if export_pdf:
        try:
            from PIL import Image
        except Exception:
            return False, "Pillow is required for PDF export"

        images = []
        for path in output_images:
            images.append(Image.open(path).convert("RGB"))
        if images:
            pdf_path = os.path.join(output_dir, "pages.pdf")
            images[0].save(pdf_path, save_all=True, append_images=images[1:])
        for img in images:
            img.close()

    return True, "Export completed"


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Transcribe a single image using local LLM.")
    parser.add_argument("--image", required=True, help="Path to an image file")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234", help="LLM base URL")
    parser.add_argument("--model", default="qwen/qwen3-vl-8b", help="LLM model name")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    parser.add_argument(
        "--prompt",
        default="general",
        choices=["general", "ja", "ja_vert"],
        help="Prompt preset",
    )
    args = parser.parse_args()

    params = ExtractParams(
        llm_base_url=args.base_url,
        llm_model=args.model,
        llm_timeout_s=args.timeout,
        llm_prompt_key=args.prompt,
    )

    try:
        text = transcribe_image_file(args.image, params)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(text)
