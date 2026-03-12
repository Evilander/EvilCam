"""CodeFormer face enhancer — ONNX-based face restoration at 512x512.

CodeFormer uses a codebook-based transformer for blind face restoration.
The ONNX model typically accepts two inputs: the face image tensor and
a fidelity weight "w" (0.0 = quality, 1.0 = fidelity to input).
If the model only has one input, the weight is omitted.
"""

from typing import Any, List
import os
import threading

import cv2
import numpy as np
import onnxruntime

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces
from modules.typing import Frame, Face
from modules.utilities import (
    is_image,
    is_video,
)
from modules.processors.frame._onnx_enhancer import (
    create_onnx_session,
    warmup_session,
    preprocess_face,
    postprocess_face,
    _get_face_affine,
    _match_grain,
    THREAD_SEMAPHORE,
)

NAME = "EVIL.FACE-ENHANCER-CODEFORMER"
INPUT_SIZE = 512
MODEL_FILE = "codeformer.onnx"

# Fidelity weight: 0.0 = max quality (more hallucination),
# 1.0 = max fidelity (closer to degraded input). 0.7 is a good default.
FIDELITY_WEIGHT = 0.7

ENHANCER = None
THREAD_LOCK = threading.Lock()

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)


def pre_check() -> bool:
    model_path = os.path.join(models_dir, MODEL_FILE)
    if not os.path.exists(model_path):
        update_status(
            f"CodeFormer ONNX model not found at {model_path}. "
            "Please place codeformer.onnx in the models folder.",
            NAME,
        )
        return False
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_enhancer() -> onnxruntime.InferenceSession:
    global ENHANCER
    with THREAD_LOCK:
        if ENHANCER is None:
            model_path = os.path.join(models_dir, MODEL_FILE)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            print(f"{NAME}: Loading ONNX model from {model_path}")
            ENHANCER = create_onnx_session(model_path)
            warmup_session(ENHANCER)
            # Log model input/output info
            for inp in ENHANCER.get_inputs():
                print(f"{NAME}: Input: {inp.name}, shape: {inp.shape}, type: {inp.type}")
            for out in ENHANCER.get_outputs():
                print(f"{NAME}: Output: {out.name}, shape: {out.shape}, type: {out.type}")
            print(f"{NAME}: Model loaded successfully.")
    return ENHANCER


def _run_codeformer(
    session: onnxruntime.InferenceSession,
    blob: np.ndarray,
    w: float | None = None,
) -> np.ndarray:
    """Run CodeFormer inference, handling both 1-input and 2-input model variants.

    Args:
        session: ONNX inference session.
        blob: Preprocessed face tensor, shape [1, 3, 512, 512], float32 in [-1, 1].
        w: Fidelity weight (only used if model has a second input).
           None = use module-level FIDELITY_WEIGHT at call time.

    Returns:
        Raw model output tensor.
    """
    if w is None:
        w = FIDELITY_WEIGHT
    inputs = session.get_inputs()
    input_feed = {inputs[0].name: blob}

    if len(inputs) >= 2:
        # Second input is the fidelity weight scalar (model requires float64)
        w_tensor = np.array([w], dtype=np.float64)
        input_feed[inputs[1].name] = w_tensor

    return session.run(None, input_feed)[0]


def _enhance_face_codeformer(
    frame: np.ndarray,
    face: Any,
    session: onnxruntime.InferenceSession,
) -> np.ndarray:
    """Enhance a single face in the frame using the CodeFormer ONNX model.

    When modules.globals.mouth_mask is enabled, the mouth region is excluded
    from enhancement to preserve the mouth handler's work.
    """
    M, inv_M = _get_face_affine(face, INPUT_SIZE)
    if M is None:
        return frame

    face_crop = cv2.warpAffine(
        frame, M, (INPUT_SIZE, INPUT_SIZE),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    blob = preprocess_face(face_crop, INPUT_SIZE)
    with THREAD_SEMAPHORE:
        output = _run_codeformer(session, blob)
    enhanced = postprocess_face(output)

    # Feathered mask for seamless blending
    mask = np.ones((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    border = max(1, INPUT_SIZE // 16)
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
        from modules.processors.frame._onnx_enhancer import _create_mouth_exclusion_mask
        mouth_excl = _create_mouth_exclusion_mask(face, frame.shape)
        if mouth_excl is not None:
            warped_mask = warped_mask * mouth_excl

    # Match grain/noise from original frame so enhanced face doesn't look "pasted on"
    warped_enhanced = _match_grain(warped_enhanced, frame, warped_mask)

    mask_3ch = warped_mask[:, :, np.newaxis]
    result = (warped_enhanced.astype(np.float32) * mask_3ch +
              frame.astype(np.float32) * (1.0 - mask_3ch))
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_face(temp_frame: Frame, face: Face) -> Frame:
    try:
        session = get_enhancer()
    except Exception as e:
        print(f"{NAME}: {e}")
        return temp_frame
    try:
        return _enhance_face_codeformer(temp_frame, face, session)
    except Exception as e:
        print(f"{NAME}: Error during face enhancement: {e}")
        return temp_frame


def process_frame(source_face: Face | None, temp_frame: Frame) -> Frame:
    if getattr(modules.globals, "many_faces", False):
        faces = get_many_faces(temp_frame)
        if faces:
            for face in faces:
                temp_frame = enhance_face(temp_frame, face)
        return temp_frame
    target_face = get_one_face(temp_frame)
    if target_face is None:
        return temp_frame
    return enhance_face(temp_frame, target_face)


def process_frame_v2(temp_frame: Frame) -> Frame:
    if getattr(modules.globals, "many_faces", False):
        faces = get_many_faces(temp_frame)
        if faces:
            for face in faces:
                temp_frame = enhance_face(temp_frame, face)
        return temp_frame
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame, target_face)
    return temp_frame


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            if progress:
                progress.update(1)
            continue
        result = process_frame(None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if progress:
            progress.update(1)


def process_image(source_path: str | None, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    if target_frame is None:
        print(f"{NAME}: Error: Failed to read target image {target_path}")
        return
    result_frame = process_frame(None, target_frame)
    cv2.imwrite(output_path, result_frame)
    print(f"{NAME}: Enhanced image saved to {output_path}")


def process_video(source_path: str | None, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
