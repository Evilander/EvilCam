"""Advanced mouth handling for undetectable face swaps.

Aggressive multi-zone mouth preservation that actually makes a visible difference:
- Strong inner mouth preservation (teeth, tongue, objects)
- Lip contour blending with wide feather zones
- Proper color matching before compositing
- Single-pass compositing (no conflicting Poisson + alpha)
- Temporal smoothing for video mode
- Occlusion-aware: gracefully handles objects near/in the mouth
- Adaptive blend weights based on color distance between original and swapped
"""

import cv2
import numpy as np
import threading
from typing import Optional, Tuple

import modules.globals
from modules.typing import Face, Frame


# Temporal smoothing state
_PREV_MOUTH_REGION = None
_PREV_MOUTH_MASK = None
_TEMPORAL_LOCK = threading.Lock()


def get_mouth_landmarks(face: Face) -> Optional[np.ndarray]:
    """Extract 106-point landmarks relevant to the mouth region.

    Includes a sanity check: if mouth landmarks are wildly outside the face
    bounding box, they're unreliable (occluded face, bad detection).
    """
    if face is None or not hasattr(face, 'landmark_2d_106'):
        return None
    lm = face.landmark_2d_106
    if lm is None or not isinstance(lm, np.ndarray) or lm.shape[0] < 106:
        return None

    # Sanity check: mouth landmarks should be within a reasonable distance
    # of the face bounding box
    if hasattr(face, 'bbox') and face.bbox is not None:
        bbox = face.bbox
        face_w = bbox[2] - bbox[0]
        face_h = bbox[3] - bbox[1]
        face_cx = (bbox[0] + bbox[2]) / 2
        face_cy = (bbox[1] + bbox[3]) / 2

        mouth_center = np.mean(lm[52:64], axis=0)
        dx = abs(mouth_center[0] - face_cx)
        dy = abs(mouth_center[1] - face_cy)

        # If mouth center is more than 1.5x face size away from face center,
        # landmarks are unreliable
        if dx > face_w * 1.5 or dy > face_h * 1.5:
            return None

    return lm


def compute_mouth_openness(landmarks: np.ndarray) -> float:
    """Compute how open the mouth is (0.0 = closed, 1.0 = wide open).

    Uses ratio of vertical mouth opening to horizontal mouth width.
    """
    upper_lip_inner = landmarks[71:77]
    lower_lip_inner = landmarks[77:83]

    upper_center = np.mean(upper_lip_inner, axis=0)
    lower_center = np.mean(lower_lip_inner, axis=0)

    vertical_gap = np.linalg.norm(lower_center - upper_center)

    mouth_left = landmarks[52]
    mouth_right = landmarks[58]
    horizontal_width = np.linalg.norm(mouth_right - mouth_left)

    if horizontal_width < 1e-6:
        return 0.0

    ratio = vertical_gap / horizontal_width
    return min(1.0, max(0.0, ratio * 2.5))


def _estimate_occlusion(
    original_frame: Frame,
    swapped_frame: Frame,
    landmarks: np.ndarray,
    frame_shape: Tuple[int, int],
) -> float:
    """Estimate how occluded the mouth region is (0.0 = clear, 1.0 = fully occluded).

    Uses two independent signals:
    1. Color variance ratio: original mouth with non-skin objects has higher HSV
       saturation variance than the clean swapped face.
    2. Edge density ratio: occluding objects (fingers, body parts, etc.) add strong
       edges that don't exist in the swap model's smooth output.

    Returns the max of both signals — either one can flag occlusion.
    """
    h, w = frame_shape
    outer_points = landmarks[52:64].astype(np.int32)

    # Expanded bounding box around mouth (20% padding for edge detection)
    x_coords = np.clip(outer_points[:, 0], 0, w - 1)
    y_coords = np.clip(outer_points[:, 1], 0, h - 1)
    x1, x2 = int(np.min(x_coords)), int(np.max(x_coords))
    y1, y2 = int(np.min(y_coords)), int(np.max(y_coords))

    pad_x = max(5, int((x2 - x1) * 0.2))
    pad_y = max(5, int((y2 - y1) * 0.2))
    x1 = max(0, x1 - pad_x)
    x2 = min(w, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(h, y2 + pad_y)

    if x2 - x1 < 10 or y2 - y1 < 10:
        return 0.0

    orig_mouth = original_frame[y1:y2, x1:x2]
    swap_mouth = swapped_frame[y1:y2, x1:x2]

    if orig_mouth.size == 0 or swap_mouth.size == 0:
        return 0.0

    # Signal 1: Color variance ratio (HSV saturation)
    orig_hsv = cv2.cvtColor(orig_mouth, cv2.COLOR_BGR2HSV).astype(np.float32)
    swap_hsv = cv2.cvtColor(swap_mouth, cv2.COLOR_BGR2HSV).astype(np.float32)

    orig_sat_std = np.std(orig_hsv[:, :, 1])
    swap_sat_std = np.std(swap_hsv[:, :, 1])

    variance_ratio = orig_sat_std / (swap_sat_std + 1e-6)
    color_signal = 0.0
    if variance_ratio > 2.0:
        color_signal = min(1.0, (variance_ratio - 2.0) / 2.5)

    # Signal 2: Edge density ratio (Canny edges)
    orig_gray = cv2.cvtColor(orig_mouth, cv2.COLOR_BGR2GRAY)
    swap_gray = cv2.cvtColor(swap_mouth, cv2.COLOR_BGR2GRAY)

    orig_edges = cv2.Canny(orig_gray, 50, 150)
    swap_edges = cv2.Canny(swap_gray, 50, 150)

    orig_edge_density = np.mean(orig_edges > 0)
    swap_edge_density = np.mean(swap_edges > 0)

    edge_ratio = orig_edge_density / (swap_edge_density + 1e-6)
    edge_signal = 0.0
    if edge_ratio > 2.5:
        edge_signal = min(1.0, (edge_ratio - 2.5) / 3.0)

    return max(color_signal, edge_signal)


def _mouth_center_and_size(landmarks: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Get mouth center, width, and height from landmarks."""
    outer_points = landmarks[52:64].astype(np.float32)
    center = np.mean(outer_points, axis=0)
    mouth_width = np.linalg.norm(landmarks[58] - landmarks[52])
    mouth_height = np.linalg.norm(
        np.mean(landmarks[77:83], axis=0) - np.mean(landmarks[71:77], axis=0)
    )
    # Use outer lip vertical extent too
    outer_height = np.linalg.norm(
        np.mean(landmarks[55:58], axis=0) - np.mean(landmarks[52:55], axis=0)
    )
    mouth_height = max(mouth_height, outer_height)
    return center, mouth_width, mouth_height


def create_inner_mouth_mask(
    landmarks: np.ndarray, frame_shape: Tuple[int, int], openness: float
) -> np.ndarray:
    """Create mask for the inner mouth (teeth/tongue/objects).

    Active even at low openness — any visible gap should be preserved.
    Much more aggressive than before.
    """
    h, w = frame_shape
    mask = np.zeros((h, w), dtype=np.float32)

    # Lower threshold — catch even slightly parted lips
    if openness < 0.02:
        return mask

    inner_points = landmarks[71:83].astype(np.float32)
    if not np.all(np.isfinite(inner_points)):
        return mask

    center = np.mean(inner_points, axis=0)

    # Scale up the inner polygon more aggressively
    # At full openness, expand to 1.5x to capture teeth edges
    scale = 1.0 + openness * 0.6
    scaled_points = (inner_points - center) * scale + center
    scaled_points = scaled_points.astype(np.int32)

    cv2.fillPoly(mask, [scaled_points], 1.0)

    # Wider feather for smoother blend
    blur_size = max(5, int(11 + openness * 30))
    blur_size = blur_size | 1  # ensure odd
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    # Full intensity — let the blend weights control the strength
    return mask


def create_outer_mouth_mask(
    landmarks: np.ndarray, frame_shape: Tuple[int, int], openness: float
) -> np.ndarray:
    """Create mask covering the lips and surrounding blending zone.

    Much wider than before — needs to cover the full lip area and transition zone.
    """
    h, w = frame_shape
    mask = np.zeros((h, w), dtype=np.float32)

    outer_points = landmarks[52:64].astype(np.float32)
    if not np.all(np.isfinite(outer_points)):
        return mask

    center = np.mean(outer_points, axis=0)

    # Expand significantly — 1.5x covers lips + chin border + surrounding skin
    expansion = 1.5 + openness * 0.3
    expanded = (outer_points - center) * expansion + center
    expanded = expanded.astype(np.int32)

    cv2.fillPoly(mask, [expanded], 1.0)

    # Wide feather for invisible transition
    blur_size = max(21, int(31 + openness * 20))
    blur_size = blur_size | 1
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    return mask


def create_chin_jaw_mask(
    landmarks: np.ndarray, frame_shape: Tuple[int, int]
) -> np.ndarray:
    """Create a mask along the jawline/chin for skin-tone blending.

    Helps smooth the transition between swapped face and original neck/jaw.
    Uses face outline landmarks 0-32.
    """
    h, w = frame_shape
    mask = np.zeros((h, w), dtype=np.float32)

    # Face outline — lower portion (chin/jaw)
    if landmarks.shape[0] < 33:
        return mask

    # Use jaw landmarks (roughly indices 0-16 in the outline)
    jaw_points = landmarks[0:17].astype(np.float32)
    if not np.all(np.isfinite(jaw_points)):
        return mask

    center = np.mean(jaw_points, axis=0)

    # Expand downward for jaw blending
    expanded = jaw_points.copy()
    expanded[:, 1] += (expanded[:, 1] - center[1]) * 0.3  # push chin points down
    expanded = expanded.astype(np.int32)

    # Create a band along the jaw
    pts = np.vstack([jaw_points.astype(np.int32), expanded[::-1]])
    cv2.fillPoly(mask, [pts], 1.0)
    mask = cv2.GaussianBlur(mask, (31, 31), 10)

    return mask


def histogram_match_region(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match color histogram of source to reference in LAB space."""
    if source.size == 0 or reference.size == 0:
        return source

    src_lab = cv2.cvtColor(source.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)

    result = np.zeros_like(src_lab)

    for i in range(3):
        src_ch = src_lab[:, :, i].ravel()
        ref_ch = ref_lab[:, :, i].ravel()

        if len(src_ch) == 0 or len(ref_ch) == 0:
            result[:, :, i] = src_lab[:, :, i]
            continue

        src_sorted = np.sort(src_ch)
        ref_sorted = np.sort(ref_ch)

        src_q = np.linspace(0, 1, len(src_sorted))
        ref_q = np.linspace(0, 1, len(ref_sorted))

        mapped = np.interp(
            src_ch, src_sorted,
            np.interp(src_q, ref_q, ref_sorted)
        )
        result[:, :, i] = mapped.reshape(src_lab.shape[:2])

    result_bgr = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return np.clip(result_bgr * 255.0, 0, 255).astype(np.uint8)


def _simple_color_transfer(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Fast mean/std color transfer in LAB space. Faster than full histogram match."""
    if source.size == 0 or reference.size == 0:
        return source

    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

    for i in range(3):
        src_mean, src_std = src_lab[:, :, i].mean(), src_lab[:, :, i].std() + 1e-6
        ref_mean, ref_std = ref_lab[:, :, i].mean(), ref_lab[:, :, i].std() + 1e-6
        src_lab[:, :, i] = (src_lab[:, :, i] - src_mean) * (ref_std / src_std) + ref_mean

    src_lab = np.clip(src_lab, 0, 255)
    return cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def temporal_smooth_mouth(
    current_mouth: np.ndarray,
    current_mask: np.ndarray,
    alpha: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Temporal smoothing to prevent mouth flickering in video."""
    global _PREV_MOUTH_REGION, _PREV_MOUTH_MASK

    with _TEMPORAL_LOCK:
        if _PREV_MOUTH_REGION is None or _PREV_MOUTH_REGION.shape != current_mouth.shape:
            _PREV_MOUTH_REGION = current_mouth.copy()
            _PREV_MOUTH_MASK = current_mask.copy()
            return current_mouth, current_mask

        smoothed_mouth = cv2.addWeighted(
            current_mouth, alpha,
            _PREV_MOUTH_REGION, 1.0 - alpha, 0
        ).astype(np.uint8)

        smoothed_mask = cv2.addWeighted(
            current_mask, alpha,
            _PREV_MOUTH_MASK, 1.0 - alpha, 0
        )

        _PREV_MOUTH_REGION = smoothed_mouth.copy()
        _PREV_MOUTH_MASK = smoothed_mask.copy()

        return smoothed_mouth, smoothed_mask


def reset_temporal_state():
    """Reset temporal smoothing state (call at start of new video/session)."""
    global _PREV_MOUTH_REGION, _PREV_MOUTH_MASK
    with _TEMPORAL_LOCK:
        _PREV_MOUTH_REGION = None
        _PREV_MOUTH_MASK = None


def _compute_color_distance(roi_a: np.ndarray, roi_b: np.ndarray) -> float:
    """Compute normalized color distance between two ROIs in LAB space.

    Returns 0.0 (identical) to 1.0 (very different).
    High distance means the original mouth likely contains non-skin content.
    """
    if roi_a.size == 0 or roi_b.size == 0:
        return 0.0

    lab_a = cv2.cvtColor(roi_a, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(roi_b, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Per-channel mean difference, weighted by perceptual importance
    # L (lightness) matters most, then a/b (chrominance)
    diff_l = abs(lab_a[:, :, 0].mean() - lab_b[:, :, 0].mean()) / 100.0
    diff_a = abs(lab_a[:, :, 1].mean() - lab_b[:, :, 1].mean()) / 128.0
    diff_b = abs(lab_a[:, :, 2].mean() - lab_b[:, :, 2].mean()) / 128.0

    return min(1.0, (diff_l * 0.5 + diff_a * 0.25 + diff_b * 0.25) * 2.0)


def process_mouth_region(
    swapped_frame: Frame,
    original_frame: Frame,
    target_face: Face,
    use_temporal_smoothing: bool = False,
    use_poisson: bool = True,
) -> Frame:
    """Main entry point for mouth handling with occlusion awareness.

    Strategy: Create a color-matched version of the original mouth, then composite
    it into the swapped frame using a multi-zone mask with wide feathering.

    Occlusion-aware: when the mouth region appears to have non-face content
    (hand, object, another person), blend weights are reduced so the swap model's
    clean output dominates — preventing gore artifacts.

    Single-pass compositing — no conflicting blend operations.

    Zone weights (clear mouth):
        Inner mouth (teeth/tongue): 95% from color-matched original
        Lip zone: 55-75% from color-matched original
        Chin/jaw zone: 25% subtle transition
    Zone weights under occlusion:
        All weights scaled down by (1 - occlusion_score)
    """
    landmarks = get_mouth_landmarks(target_face)
    if landmarks is None:
        return swapped_frame

    frame_shape = swapped_frame.shape[:2]
    openness = compute_mouth_openness(landmarks)

    # Estimate occlusion — are there non-face objects near the mouth?
    occlusion = _estimate_occlusion(
        original_frame, swapped_frame, landmarks, frame_shape
    )

    # If heavily occluded, the swap model's output is safer than the original
    if occlusion > 0.85:
        return swapped_frame

    # Create zone masks
    inner_mask = create_inner_mouth_mask(landmarks, frame_shape, openness)
    outer_mask = create_outer_mouth_mask(landmarks, frame_shape, openness)
    chin_mask = create_chin_jaw_mask(landmarks, frame_shape)

    # Combined mask for ROI extraction (include chin zone for smoother jaw blending)
    combined_mask = np.maximum(inner_mask, np.maximum(outer_mask, chin_mask))
    y_idx, x_idx = np.where(combined_mask > 0.005)

    if len(x_idx) == 0 or len(y_idx) == 0:
        return swapped_frame

    # Generous padding for the ROI
    pad = 20
    h, w = frame_shape
    x_min = max(0, np.min(x_idx) - pad)
    x_max = min(w, np.max(x_idx) + pad)
    y_min = max(0, np.min(y_idx) - pad)
    y_max = min(h, np.max(y_idx) + pad)

    if (x_max - x_min) < 15 or (y_max - y_min) < 15:
        return swapped_frame

    # Extract ROIs
    orig_roi = original_frame[y_min:y_max, x_min:x_max].copy()
    swap_roi = swapped_frame[y_min:y_max, x_min:x_max].copy()
    inner_roi = inner_mask[y_min:y_max, x_min:x_max]
    outer_roi = outer_mask[y_min:y_max, x_min:x_max]
    chin_roi = chin_mask[y_min:y_max, x_min:x_max]

    if orig_roi.size == 0 or swap_roi.size == 0:
        return swapped_frame

    # Color distance between original and swapped mouth — high means non-skin content
    color_dist = _compute_color_distance(orig_roi, swap_roi)

    # Color-match original mouth to swapped face skin tone
    color_matched = _simple_color_transfer(orig_roi, swap_roi)

    # Occlusion-aware weight scaling
    # occlusion: 0.0 = clear mouth, 1.0 = fully occluded
    # color_dist: 0.0 = similar colors, 1.0 = very different (foreign object)
    # Combine both signals: either one can reduce preservation
    suppression = max(occlusion, color_dist * 0.6)
    weight_scale = 1.0 - suppression

    # Build composite blend mask with zone weights
    # Inner zone: strong preservation (teeth, tongue) — scaled by occlusion
    inner_weight = (0.95 if openness > 0.02 else 0.0) * weight_scale

    # Outer/lip zone: moderate preservation (expression, lip shape)
    lip_weight = (0.55 + openness * 0.2) * weight_scale

    # Chin/jaw zone: subtle blending for skin-tone transition
    chin_weight = 0.25 * weight_scale

    # Compute the final per-pixel weight map
    outer_only = np.maximum(0, outer_roi - inner_roi)
    chin_only = np.maximum(0, chin_roi - np.maximum(inner_roi, outer_roi))
    blend_map = (inner_roi * inner_weight) + (outer_only * lip_weight) + (chin_only * chin_weight)
    blend_map = np.clip(blend_map, 0.0, 1.0)

    # Apply temporal smoothing before compositing
    if use_temporal_smoothing:
        color_matched, blend_map = temporal_smooth_mouth(
            color_matched, blend_map, alpha=0.75
        )

    # Single-pass composite: swapped * (1 - blend) + color_matched * blend
    blend_3ch = blend_map[:, :, np.newaxis].astype(np.float32)
    composited = (
        swap_roi.astype(np.float32) * (1.0 - blend_3ch) +
        color_matched.astype(np.float32) * blend_3ch
    )
    composited = np.clip(composited, 0, 255).astype(np.uint8)

    # Write composited ROI back
    result = swapped_frame.copy()
    result[y_min:y_max, x_min:x_max] = composited

    # Poisson blending: skip entirely when occlusion is significant (prevents gore),
    # use NORMAL_CLONE (gentler) for moderate cases, MIXED_CLONE only when clear
    if use_poisson and suppression < 0.4 and np.max(blend_map) > 0.1:
        poisson_mask = (blend_map * 255).astype(np.uint8)
        _, poisson_mask = cv2.threshold(poisson_mask, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        poisson_mask = cv2.erode(poisson_mask, kernel, iterations=1)

        poisson_y, poisson_x = np.where(poisson_mask > 0)
        if len(poisson_x) > 10 and len(poisson_y) > 10:
            cx = int((np.min(poisson_x) + np.max(poisson_x)) / 2) + x_min
            cy = int((np.min(poisson_y) + np.max(poisson_y)) / 2) + y_min

            if 0 < cx < w and 0 < cy < h:
                full_mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
                full_mask[:poisson_mask.shape[0], :poisson_mask.shape[1]] = poisson_mask

                # Use gentler NORMAL_CLONE when any occlusion signal is present
                clone_mode = cv2.NORMAL_CLONE if suppression > 0.1 else cv2.MIXED_CLONE

                try:
                    result = cv2.seamlessClone(
                        composited, result, full_mask,
                        (cx, cy), clone_mode
                    )
                except Exception:
                    pass  # alpha composite already applied, Poisson is optional polish

    return result
