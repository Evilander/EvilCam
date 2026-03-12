"""Shared ONNX-based face enhancement utilities for GPEN-BFR models.

Provides session creation, pre/post processing, and the core
enhance-face-via-ONNX pipeline.

Mouth-aware enhancement: when modules.globals.mouth_mask is enabled,
the enhancement blend mask excludes the mouth region so enhancers
don't destroy the mouth handler's careful preservation work.
"""

import os
import platform
import threading
from typing import Any, Optional

import cv2
import numpy as np
import onnxruntime

import modules.globals

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# Limit concurrent ONNX calls to avoid VRAM exhaustion on multi-face frames
THREAD_SEMAPHORE = threading.Semaphore(min(max(1, (os.cpu_count() or 1)), 8))


def create_onnx_session(model_path: str) -> onnxruntime.InferenceSession:
    """Create an ONNX Runtime session using the configured execution providers."""
    providers = modules.globals.execution_providers
    session = onnxruntime.InferenceSession(model_path, providers=providers)
    return session


def warmup_session(session: onnxruntime.InferenceSession) -> None:
    """Run a dummy inference pass to trigger JIT / compile caching."""
    try:
        _onnx_to_np = {
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
        }
        input_feed = {
            inp.name: np.zeros(
                [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape],
                dtype=_onnx_to_np.get(inp.type, np.float32),
            )
            for inp in session.get_inputs()
        }
        session.run(None, input_feed)
    except Exception as e:
        print(f"ONNX enhancer warmup skipped (non-fatal): {e}")


def preprocess_face(face_img: np.ndarray, input_size: int) -> np.ndarray:
    """Resize, normalize, and convert a BGR face crop to ONNX input blob.

    GPEN-BFR expects [1, 3, H, W] float32 in RGB, normalized to [-1, 1].
    """
    resized = cv2.resize(face_img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0 * 2.0 - 1.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return blob


def postprocess_face(output: np.ndarray) -> np.ndarray:
    """Convert ONNX output [1, 3, H, W] float32 back to BGR uint8 image."""
    img = output[0].transpose(1, 2, 0)
    img = ((img + 1.0) / 2.0 * 255.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _get_face_affine(face: Any, input_size: int):
    """Compute affine transform to align a face to GPEN input space.

    Returns (M, inv_M) — forward and inverse affine matrices.
    """
    template = np.array([
        [0.31556875, 0.4615741],
        [0.68262291, 0.4615741],
        [0.50009375, 0.6405054],
        [0.34947187, 0.8246919],
        [0.65343645, 0.8246919],
    ], dtype=np.float32) * input_size

    landmarks = None
    if hasattr(face, "kps") and face.kps is not None:
        landmarks = face.kps.astype(np.float32)
    elif hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        lm106 = face.landmark_2d_106
        landmarks = np.array([
            lm106[38],  # left eye
            lm106[88],  # right eye
            lm106[86],  # nose tip
            lm106[52],  # left mouth
            lm106[61],  # right mouth
        ], dtype=np.float32)

    if landmarks is None or len(landmarks) < 5:
        return None, None

    M = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)[0]
    if M is None:
        return None, None
    inv_M = cv2.invertAffineTransform(M)
    return M, inv_M


def _create_mouth_exclusion_mask(
    face: Any,
    frame_shape: tuple,
    feather: int = 25,
) -> Optional[np.ndarray]:
    """Create a mask that EXCLUDES the mouth region from enhancement.

    Returns a float32 mask where 0.0 = mouth (don't enhance), 1.0 = rest of face (enhance).
    Returns None if landmarks aren't available.
    """
    if not hasattr(face, 'landmark_2d_106') or face.landmark_2d_106 is None:
        return None

    lm = face.landmark_2d_106
    if not isinstance(lm, np.ndarray) or lm.shape[0] < 106:
        return None

    h, w = frame_shape[:2]

    # Use outer mouth landmarks (52-63) + inner (71-82) for a generous mouth region
    outer_points = lm[52:64].astype(np.float32)
    inner_points = lm[71:83].astype(np.float32)

    if not np.all(np.isfinite(outer_points)):
        return None

    center = np.mean(outer_points, axis=0)

    # Expand the outer mouth polygon generously to cover the full lip area
    # + surrounding skin transition zone. 1.6x covers lips + chin border.
    expansion = 1.6
    expanded = (outer_points - center) * expansion + center
    expanded = expanded.astype(np.int32)

    # Create the exclusion mask (start with all 1s = enhance everything)
    mask = np.ones((h, w), dtype=np.float32)

    # Fill the mouth region with 0 (don't enhance)
    cv2.fillPoly(mask, [expanded], 0.0)

    # Wide feather for smooth transition between enhanced/non-enhanced regions
    blur_k = feather * 2 + 1
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

    return mask


def _match_grain(
    enhanced: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Add noise/grain to the enhanced face to match the original frame's texture.

    Enhancement models output perfectly smooth faces that look fake when pasted
    onto noisy/grainy video. This estimates the original's noise profile from
    a peripheral ring around the face and adds temporally-varying grain.

    Args:
        enhanced: The enhanced face warped back to frame space (uint8).
        original: The original frame (uint8).
        mask: The blend mask (float32, 0-1) showing where enhancement is applied.

    Returns:
        Enhanced face with matching grain added (uint8).
    """
    active_2d = mask if mask.ndim == 2 else mask[:, :, 0]
    if not np.any(active_2d > 0.1):
        return enhanced

    # Estimate noise from a ring AROUND the face (not inside — that's already swapped).
    # Dilate the mask to get a peripheral sampling zone.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    dilated = cv2.dilate((active_2d > 0.1).astype(np.uint8), kernel, iterations=1)
    peripheral = (dilated > 0) & (active_2d < 0.1)

    if np.sum(peripheral) < 100:
        # Not enough peripheral pixels — fall back to whole-frame noise estimate
        peripheral = active_2d < 0.1

    if np.sum(peripheral) < 100:
        return enhanced

    # High-pass filter to extract noise (original - blurred)
    original_blur = cv2.GaussianBlur(original, (5, 5), 0)
    noise = original.astype(np.float32) - original_blur.astype(np.float32)

    # Compute per-channel noise std from the peripheral ring
    noise_stds = []
    for c in range(min(3, noise.shape[2]) if noise.ndim == 3 else 1):
        ch = noise[:, :, c] if noise.ndim == 3 else noise
        noise_stds.append(float(np.std(ch[peripheral])))

    avg_std = np.mean(noise_stds)
    if avg_std < 1.0:
        return enhanced

    # Seed from frame content for temporally-varying but deterministic grain.
    # Different frames produce different noise patterns (no static overlay).
    content_hash = int(np.sum(original[::16, ::16, :1].astype(np.uint32))) & 0x7FFFFFFF
    rng = np.random.RandomState(content_hash)

    # Generate per-channel noise matching each channel's statistics
    if enhanced.ndim == 3:
        synthetic_noise = np.stack([
            rng.normal(0, max(s, 0.5), enhanced.shape[:2]).astype(np.float32)
            for s in noise_stds
        ], axis=-1)
    else:
        synthetic_noise = rng.normal(0, avg_std, enhanced.shape).astype(np.float32)

    # Apply noise scaled by the blend mask (smooth transition at edges)
    mask_3ch = active_2d[:, :, np.newaxis] if enhanced.ndim == 3 else active_2d
    result = enhanced.astype(np.float32) + synthetic_noise * mask_3ch
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_face_onnx(
    frame: np.ndarray,
    face: Any,
    session: onnxruntime.InferenceSession,
    input_size: int,
) -> np.ndarray:
    """Enhance a single face in the frame using an ONNX face restoration model.

    When modules.globals.mouth_mask is enabled, the mouth region is excluded
    from enhancement to preserve the mouth handler's work.
    """
    M, inv_M = _get_face_affine(face, input_size)
    if M is None:
        return frame

    face_crop = cv2.warpAffine(
        frame, M, (input_size, input_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    blob = preprocess_face(face_crop, input_size)
    with THREAD_SEMAPHORE:
        output = session.run(None, {session.get_inputs()[0].name: blob})[0]
    enhanced = postprocess_face(output)

    # Create mask for blending (feathered edges)
    mask = np.ones((input_size, input_size), dtype=np.float32)
    border = max(1, input_size // 16)
    mask[:border, :] = np.linspace(0, 1, border)[:, np.newaxis]
    mask[-border:, :] = np.linspace(1, 0, border)[:, np.newaxis]
    mask[:, :border] = np.minimum(mask[:, :border], np.linspace(0, 1, border)[np.newaxis, :])
    mask[:, -border:] = np.minimum(mask[:, -border:], np.linspace(1, 0, border)[np.newaxis, :])

    h, w = frame.shape[:2]
    warped_enhanced = cv2.warpAffine(
        enhanced, inv_M, (w, h),
        flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        mask, inv_M, (w, h),
        flags=cv2.INTER_LINEAR, borderValue=0,
    )

    # Mouth-aware enhancement: exclude mouth region from the blend mask
    if getattr(modules.globals, "mouth_mask", False):
        mouth_excl = _create_mouth_exclusion_mask(face, frame.shape)
        if mouth_excl is not None:
            warped_mask = warped_mask * mouth_excl

    # Match grain/noise from original frame so enhanced face doesn't look "pasted on"
    warped_enhanced = _match_grain(warped_enhanced, frame, warped_mask)

    mask_3ch = warped_mask[:, :, np.newaxis]
    result = (warped_enhanced.astype(np.float32) * mask_3ch +
              frame.astype(np.float32) * (1.0 - mask_3ch))
    return np.clip(result, 0, 255).astype(np.uint8)
