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

from .models import PageItem, PageResult, RawFrame
from .prompts import get_prompt


@dataclass
class ExtractParams:
    analysis_fps: float = 1.0
    analysis_long_side: int = 0
    rotation_degrees: int = 0
    max_interval_s: float = 5.0
    llm_base_url: str = "http://127.0.0.1:1234"
    llm_model: str = "qwen/qwen3-vl-8b"
    llm_timeout_s: int = 180
    llm_prompt_key: str = "general"
    llm_split_4: bool = True
    llm_max_tokens: int = 1024
    or_window: int = 6
    min_peak_distance_s: float = 0.12
    peak_mad_k: float = 3.0
    boundary_motion_weight: float = 0.5
    boundary_change_weight: float = 0.5
    search_start_frames: int = 1
    search_end_s: float = 0.4
    output_per_peak: int = 1
    use_quad_in_score: bool = False
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
    if long_side <= 0:
        return frame
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
    analysis_fps: float,
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

    safe_analysis_fps = max(0.01, float(analysis_fps))
    step = max(1, int(round(fps / safe_analysis_fps)))
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

def _split_frame_vertical(frame: np.ndarray, parts: int = 4, right_to_left: bool = True) -> List[np.ndarray]:
    h, w = frame.shape[:2]
    if parts <= 1 or w <= 1:
        return [frame]
    boundaries = [int(round(i * w / parts)) for i in range(parts + 1)]
    slices = [frame[:, boundaries[i] : boundaries[i + 1]] for i in range(parts)]
    if right_to_left:
        slices.reverse()
    return slices


def _split_frame_quadrants(frame: np.ndarray) -> List[np.ndarray]:
    h, w = frame.shape[:2]
    if h <= 1 or w <= 1:
        return [frame]
    mid_h = h // 2
    mid_w = w // 2
    return [
        frame[0:mid_h, 0:mid_w],
        frame[0:mid_h, mid_w:w],
        frame[mid_h:h, 0:mid_w],
        frame[mid_h:h, mid_w:w],
    ]


def _prepare_llm_payload(frame: np.ndarray, params: ExtractParams, max_tokens: int) -> Dict[str, object]:
    if frame.ndim == 2:
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        bgr = frame
    ok, buffer = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame for LLM")
    image_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    prompt = get_prompt(params.llm_prompt_key)
    return {
        "model": params.llm_model,
        "max_tokens": max_tokens,
        "temperature": 0.7,
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


def _llm_max_tokens(params: ExtractParams, split: bool) -> int:
    if split:
        return max(64, int(params.llm_max_tokens // 4))
    return int(params.llm_max_tokens)


def _parse_llm_response(response: Dict[str, object]) -> str:
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


def _transcribe_frame_once(frame: np.ndarray, params: ExtractParams, max_tokens: int) -> str:
    payload = _prepare_llm_payload(frame, params, max_tokens)
    response = _llm_request(params.llm_base_url, payload, params.llm_timeout_s)
    return _parse_llm_response(response)


def transcribe_frame(frame: np.ndarray, params: ExtractParams) -> str:
    should_split = params.llm_split_4 and "vert" in params.llm_prompt_key
    if not should_split:
        max_tokens = _llm_max_tokens(params, split=False)
        return _transcribe_frame_once(frame, params, max_tokens)

    max_tokens = _llm_max_tokens(params, split=True)
    parts = _split_frame_vertical(frame)
    texts = [_transcribe_frame_once(part, params, max_tokens) for part in parts]
    return "\n\n\n".join(texts).strip()


def transcribe_image_file(image_path: str, params: ExtractParams) -> str:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return transcribe_frame(image, params)


def compute_text_change_series(
    video_path: str,
    infos: List[Dict[str, int]],
    params: ExtractParams,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    output_dir: Optional[str] = None,
    on_item: Optional[Callable[[PageItem], None]] = None,
) -> Tuple[List[str], List[str], List[float], List[float], List[RawFrame]]:
    if not infos:
        return [], [], [], [], []

    texts_raw: List[str] = []
    texts_norm: List[str] = []
    raw_frames: List[RawFrame] = []
    total = len(infos)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video for LLM transcription")

    pages_dir = None
    if output_dir:
        pages_dir = os.path.join(output_dir, "pages")
        ensure_dir(pages_dir)

    try:
        for idx, info in enumerate(infos):
            if should_stop and should_stop():
                break
            frame = extract_highres_frame(
                cap,
                frame_index=info.get("frame_index"),
                time_ms=info.get("time_ms"),
                rotation_degrees=params.rotation_degrees,
            )
            image_path = ""
            if pages_dir is not None:
                image_path = os.path.join(pages_dir, f"page_{idx + 1:04d}.png")
                cv2.imwrite(image_path, frame)
                if on_item:
                    on_item(
                        PageItem(
                            image_path=image_path,
                            timestamp_ms=int(info.get("time_ms", 0)),
                            analysis_index=idx,
                            is_selected=None,
                            ocr_text=None,
                        )
                    )
            text = transcribe_frame(frame, params)
            text_value = text.strip()
            texts_raw.append(text_value)
            texts_norm.append(_normalize_text(text_value))
            if pages_dir is not None:
                text_path = os.path.join(pages_dir, f"page_{idx + 1:04d}.txt")
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(text_value)
                raw_frames.append(
                    RawFrame(
                        analysis_index=idx,
                        source_frame_index=int(info.get("frame_index", -1)),
                        timestamp_ms=int(info.get("time_ms", 0)),
                        image_path=image_path,
                        ocr_text=text_value,
                    )
                )
                _write_progress_json(output_dir, video_path, raw_frames, [])
                if on_item:
                    on_item(
                        PageItem(
                            image_path=image_path,
                            timestamp_ms=int(info.get("time_ms", 0)),
                            analysis_index=idx,
                            is_selected=None,
                            ocr_text=text_value,
                        )
                    )
            if progress_cb:
                percent = 20 + int(((idx + 1) / max(1, total)) * 30)
                progress_cb(percent, f"LLM {idx + 1}/{total}")
    finally:
        cap.release()

    change_raw = [0.0]
    for i in range(1, len(texts_norm)):
        sim = _text_similarity(texts_norm[i], texts_norm[i - 1])
        change_raw.append(1.0 - sim)

    change_smooth = _smooth_series(change_raw, window=5)
    return texts_raw, texts_norm, change_raw, change_smooth, raw_frames


def segment_by_two_refs(
    sim: Callable[[int, int], float],
    n: int,
    thr: float,
    n_trans_min: int,
    n_trans_max: int,
) -> List[int]:
    if n <= 1:
        return []

    l_step = max(1, n_trans_min // 2)
    k_confirm = max(1, l_step // 4)
    m_grid = 16
    boundaries: List[int] = []
    p = 0

    def g(t: int, left: int, right: int) -> float:
        return sim(t, left) - sim(t, right)

    while True:
        q = p + l_step
        if q >= n:
            break

        while q < n and sim(p, q) > thr and (q - p) < n_trans_max:
            q += l_step

        if q >= n:
            break

        forced = False
        if (q - p) >= n_trans_max and sim(p, q) > thr:
            q = min(p + n_trans_max, n - 1)
            forced = True
        if not forced:
            while q + k_confirm < n:
                mx = max(sim(p, q + i) for i in range(k_confirm))
                if mx <= thr:
                    break
                q += 1
                if (q - p) >= n_trans_max:
                    q = min(p + n_trans_max, n - 1)
                    forced = True
                    break

        if q <= p:
            break

        best_t = p
        best_val = float("inf")
        for i in range(1, m_grid):
            t = p + (q - p) * i // m_grid
            val = abs(g(t, p, q))
            if val < best_val:
                best_val = val
                best_t = t

        lo = max(p, best_t - l_step // 8)
        hi = min(q, best_t + l_step // 8)
        boundary = best_t
        best_val = float("inf")
        for t in range(lo, hi + 1):
            val = abs(g(t, p, q))
            if val < best_val:
                best_val = val
                boundary = t

        if q == p + 1:
            boundary = q

        boundaries.append(boundary)
        p = q

    return boundaries


def segment_by_two_refs_texts(
    texts_norm: List[str],
    thr: float,
    n_trans_min: int,
    n_trans_max: int,
) -> List[int]:
    return segment_by_two_refs(
        lambda i, j: _text_similarity(texts_norm[i], texts_norm[j]),
        len(texts_norm),
        thr,
        n_trans_min,
        n_trans_max,
    )


def _expand_segment_starts(
    boundaries: List[int],
    total: int,
    n_trans_max: int,
) -> List[int]:
    segment_starts = [0]
    for boundary in boundaries:
        if boundary > segment_starts[-1]:
            segment_starts.append(boundary)

    expanded_starts = list(segment_starts)
    for idx, start in enumerate(segment_starts):
        end = segment_starts[idx + 1] if idx + 1 < len(segment_starts) else total
        current = start
        while (end - current) > n_trans_max:
            current += n_trans_max
            if current < end:
                expanded_starts.append(current)

    return sorted(set(expanded_starts))


def _select_analysis_indices(
    boundaries: List[int],
    total: int,
    n_trans_max: int,
) -> List[int]:
    if total <= 0:
        return []
    if not boundaries:
        return list(range(total))

    segment_starts = _expand_segment_starts(boundaries, total, n_trans_max)
    indices: List[int] = []
    for idx, start in enumerate(segment_starts):
        end = segment_starts[idx + 1] if idx + 1 < len(segment_starts) else total
        if end <= start:
            continue
        mid = start + (end - start) // 2
        indices.append(mid)
    return indices


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
        indices = list(range(start, end + 1))

        for idx in indices:
            sharp = _laplacian_variance(frames[idx])
            motion = float(motion_series[idx])
            sharp_vals.append(sharp)
            motion_vals.append(motion)

        sharp_norm = _normalize(sharp_vals)
        motion_norm = _normalize(motion_vals)

        scores: List[Tuple[int, float]] = []
        for i, idx in enumerate(indices):
            score = sharp_norm[i] - params.motion_weight * motion_norm[i]
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
                    "quad_score": 0.0,
                },
            }
        )
    return selections


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
    if not ok and frame_index is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    if not ok and time_ms is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        ok, frame = cap.read()

    if not ok:
        raise RuntimeError("Failed to decode high-res frame")
    return rotate_frame(frame, rotation_degrees)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _add_sim_prev(raw_items: List[Dict[str, object]]) -> None:
    prev_norm: Optional[str] = None
    for item in raw_items:
        text_value = item.get("ocr_text") or ""
        norm = _normalize_text(text_value) if text_value else ""
        if prev_norm is None:
            item["sim_prev"] = None
        else:
            item["sim_prev"] = _text_similarity(norm, prev_norm)
        prev_norm = norm


def _write_json_payloads(
    output_dir: str,
    source_video_path: str,
    raw_items: List[Dict[str, object]],
    selected_items: List[Dict[str, object]],
) -> None:
    raw_payload = {"source_video_path": source_video_path, "items": raw_items}
    selected_payload = {"source_video_path": source_video_path, "items": selected_items}
    with open(os.path.join(output_dir, "pages_raw.json"), "w", encoding="utf-8") as f:
        json.dump(raw_payload, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "pages_selected.json"), "w", encoding="utf-8") as f:
        json.dump(selected_payload, f, ensure_ascii=False, indent=2)


def _write_progress_json(
    output_dir: str,
    source_video_path: str,
    raw_frames: List[RawFrame],
    results: List[PageResult],
) -> None:
    raw_items: List[Dict[str, object]] = []
    raw_index_map: Dict[int, Dict[str, object]] = {}
    for idx, raw in enumerate(raw_frames, start=1):
        rel_image = os.path.relpath(raw.image_path, output_dir)
        entry = {
            "raw_index": idx,
            "analysis_index": raw.analysis_index,
            "source_frame_index": raw.source_frame_index,
            "timestamp_ms": raw.timestamp_ms,
            "image": rel_image,
            "ocr_text": raw.ocr_text,
        }
        raw_items.append(entry)
        raw_index_map[raw.analysis_index] = entry

    _add_sim_prev(raw_items)

    selected_items: List[Dict[str, object]] = []
    for idx, result in enumerate(results, start=1):
        image_entry = raw_index_map.get(result.analysis_index)
        if image_entry is None:
            rel_image = os.path.relpath(result.image_path, output_dir)
        else:
            rel_image = image_entry["image"]
        selected_items.append(
            {
                "selected_index": idx,
                "analysis_index": result.analysis_index,
                "timestamp_ms": result.timestamp_ms,
                "image": rel_image,
                "ocr_text": result.ocr_text,
                "score_components": result.score_components,
                "warp_applied": result.warp_applied,
                "quad_points": result.quad_points,
            }
        )

    _write_json_payloads(output_dir, source_video_path, raw_items, selected_items)


def extract_pages(
    video_path: str,
    params: ExtractParams,
    output_dir: Optional[str] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    on_item: Optional[Callable[[PageItem], None]] = None,
) -> Tuple[str, List[PageResult], List[RawFrame]]:
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="speedread_")

    ensure_dir(output_dir)
    if progress_cb:
        progress_cb(5, "Decoding analysis frames")
    _frames, infos, video_info = decode_analysis_frames(
        video_path,
        params.analysis_fps,
        params.analysis_long_side,
        rotation_degrees=params.rotation_degrees,
        progress_cb=progress_cb,
        should_stop=should_stop,
    )

    if should_stop and should_stop():
        return output_dir, [], []

    if progress_cb:
        progress_cb(20, "Running local LLM transcription (hi-res)")
    texts, texts_norm, _change_raw, change_smooth, raw_frames = compute_text_change_series(
        video_path,
        infos,
        params,
        progress_cb=progress_cb,
        should_stop=should_stop,
        output_dir=output_dir,
        on_item=on_item,
    )

    if should_stop and should_stop():
        return output_dir, [], []

    fps = max(0.01, float(params.analysis_fps))
    n_trans_min = 1
    n_trans_max = max(1, int(round(params.max_interval_s * fps)))
    boundaries = segment_by_two_refs_texts(
        texts_norm,
        thr=0.2,
        n_trans_min=n_trans_min,
        n_trans_max=n_trans_max,
    )

    if progress_cb:
        progress_cb(50, f"Selecting frames ({len(boundaries)})")

    selections: List[Dict[str, object]] = []
    if not boundaries and progress_cb:
        progress_cb(55, "No text segments found, exporting all frames")

    selected_indices = _select_analysis_indices(boundaries, len(texts_norm), n_trans_max)
    if not selected_indices and progress_cb:
        progress_cb(55, "No frames selected, exporting all frames")

    for mid in selected_indices:
        score_val = change_smooth[mid] if mid < len(change_smooth) else 0.0
        selections.append(
            {
                "analysis_index": mid,
                "pce_peak_index": mid,
                "score_components": {"text_change": float(score_val)},
            }
        )

    results: List[PageResult] = []
    selection_map = {sel["analysis_index"]: sel for sel in selections}
    for idx, raw in enumerate(raw_frames):
        sel = selection_map.get(raw.analysis_index)
        if sel is not None:
            result = PageResult(
                image_path=raw.image_path,
                timestamp_ms=raw.timestamp_ms,
                analysis_index=raw.analysis_index,
                pce_peak_index=int(sel["pce_peak_index"]),
                score_components=sel["score_components"],
                warp_applied=False,
                quad_points=None,
                ocr_text=raw.ocr_text,
            )
            results.append(result)
        if progress_cb:
            percent = 60 + int(((idx + 1) / max(1, len(raw_frames))) * 30)
            progress_cb(percent, f"Prepared {idx + 1}/{len(raw_frames)}")

    if on_item:
        for raw in raw_frames:
            on_item(
                PageItem(
                    image_path=raw.image_path,
                    timestamp_ms=raw.timestamp_ms,
                    analysis_index=raw.analysis_index,
                    is_selected=raw.analysis_index in selection_map,
                    ocr_text=raw.ocr_text,
                )
            )

    _write_progress_json(output_dir, video_path, raw_frames, results)
    if progress_cb:
        progress_cb(95, "Finalizing")

    return output_dir, results, raw_frames


def extract_single_frame(
    video_path: str,
    time_ms: int,
    params: ExtractParams,
    output_dir: str,
    index: int,
) -> PageResult:
    ensure_dir(output_dir)
    pages_dir = os.path.join(output_dir, "pages")
    ensure_dir(pages_dir)

    cap = cv2.VideoCapture(video_path)
    frame = extract_highres_frame(cap, time_ms=time_ms, rotation_degrees=params.rotation_degrees)
    cap.release()

    quad_points = None
    warp_applied = False


    image_path = os.path.join(pages_dir, f"page_{index:04d}.png")
    cv2.imwrite(image_path, frame)

    return PageResult(
        image_path=image_path,
        timestamp_ms=int(time_ms),
        analysis_index=-1,
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
    raw_frames: Optional[List[RawFrame]] = None,
) -> Tuple[bool, str]:
    pages_dir = os.path.join(output_dir, "pages")
    ensure_dir(pages_dir)

    output_images: List[str] = []
    raw_items: List[Dict[str, object]] = []
    selected_items: List[Dict[str, object]] = []
    raw_index_map: Dict[int, Dict[str, object]] = {}

    if raw_frames:
        for idx, raw in enumerate(raw_frames, start=1):
            dst_image = os.path.join(pages_dir, f"page_{idx:04d}.png")
            if os.path.abspath(raw.image_path) != os.path.abspath(dst_image):
                with open(raw.image_path, "rb") as src_f:
                    with open(dst_image, "wb") as dst_f:
                        dst_f.write(src_f.read())
            output_images.append(dst_image)
            rel_image = os.path.relpath(dst_image, output_dir)
            entry = {
                "raw_index": idx,
                "analysis_index": raw.analysis_index,
                "source_frame_index": raw.source_frame_index,
                "timestamp_ms": raw.timestamp_ms,
                "image": rel_image,
                "ocr_text": raw.ocr_text,
            }
            raw_items.append(entry)
            raw_index_map[raw.analysis_index] = entry
    else:
        for idx, result in enumerate(results, start=1):
            dst_image = os.path.join(pages_dir, f"page_{idx:04d}.png")
            if os.path.abspath(result.image_path) != os.path.abspath(dst_image):
                with open(result.image_path, "rb") as src_f:
                    with open(dst_image, "wb") as dst_f:
                        dst_f.write(src_f.read())
            output_images.append(dst_image)
            rel_image = os.path.relpath(dst_image, output_dir)
            entry = {
                "raw_index": idx,
                "analysis_index": result.analysis_index,
                "source_frame_index": -1,
                "timestamp_ms": result.timestamp_ms,
                "image": rel_image,
                "ocr_text": result.ocr_text,
            }
            raw_items.append(entry)
            raw_index_map[result.analysis_index] = entry

    _add_sim_prev(raw_items)

    next_image_index = len(output_images) + 1
    for idx, result in enumerate(results, start=1):
        image_entry = raw_index_map.get(result.analysis_index)
        if image_entry is None:
            dst_image = os.path.join(pages_dir, f"page_{next_image_index:04d}.png")
            if os.path.abspath(result.image_path) != os.path.abspath(dst_image):
                with open(result.image_path, "rb") as src_f:
                    with open(dst_image, "wb") as dst_f:
                        dst_f.write(src_f.read())
            output_images.append(dst_image)
            image_entry = {"image": os.path.relpath(dst_image, output_dir)}
            next_image_index += 1
        selected_items.append(
            {
                "selected_index": idx,
                "analysis_index": result.analysis_index,
                "timestamp_ms": result.timestamp_ms,
                "image": image_entry["image"],
                "ocr_text": result.ocr_text,
                "score_components": result.score_components,
                "warp_applied": result.warp_applied,
                "quad_points": result.quad_points,
            }
        )

    _write_json_payloads(output_dir, source_video_path, raw_items, selected_items)

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
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens per request")
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
        llm_max_tokens=args.max_tokens,
    )

    try:
        text = transcribe_image_file(args.image, params)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(text)
