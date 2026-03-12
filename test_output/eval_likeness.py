"""Likeness evaluation: compare face swap quality across pipeline configurations.

Runs the same frame through multiple swap+enhancer combos and measures:
  - Identity score: ArcFace cosine similarity to source face (Jaimee)
  - Sharpness: Laplacian variance of the face region (higher = sharper)
  - Color fidelity: LAB color distance between swapped face and original target

Outputs a ranked table + visual comparison grid.
"""

import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import onnxruntime

import modules.globals

# Configure GPU
available = onnxruntime.get_available_providers()
if "CUDAExecutionProvider" in available:
    modules.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
elif "DmlExecutionProvider" in available:
    modules.globals.execution_providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
else:
    modules.globals.execution_providers = ["CPUExecutionProvider"]

modules.globals.execution_threads = 1
modules.globals.max_memory = 16

print(f"Execution providers: {modules.globals.execution_providers}")

from modules.face_analyser import get_one_face, get_many_faces

# Paths
SOURCE_PATH = r"B:\projects\archive\jema\IMG_0027.jpg"
VIDEO_PATH = os.path.join(PROJECT_ROOT, "test_output", "redgifs_test_bbc.mp4")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_output", "eval_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_frame(video_path, pct=0.5):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * pct))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def compute_identity_score(source_face, result_frame):
    """ArcFace cosine similarity between source and result face."""
    result_face = get_one_face(result_frame)
    if result_face is None or result_face.normed_embedding is None:
        return 0.0
    sim = float(np.dot(source_face.normed_embedding, result_face.normed_embedding))
    return max(0.0, min(1.0, sim))


def compute_face_sharpness(frame, face=None):
    """Laplacian variance of the face region — higher = sharper."""
    if face is not None and hasattr(face, 'bbox') and face.bbox is not None:
        bbox = face.bbox.astype(int)
        x1, y1 = max(0, bbox[0]), max(0, bbox[1])
        x2, y2 = min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])
        roi = frame[y1:y2, x1:x2]
    else:
        # Use center region as proxy
        h, w = frame.shape[:2]
        roi = frame[h//4:3*h//4, w//4:3*w//4]

    if roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_color_fidelity(original_frame, result_frame, face):
    """LAB color distance in the face region — lower = better color match."""
    if face is None or not hasattr(face, 'bbox'):
        return 1.0
    bbox = face.bbox.astype(int)
    x1, y1 = max(0, bbox[0]), max(0, bbox[1])
    x2, y2 = min(original_frame.shape[1], bbox[2]), min(original_frame.shape[0], bbox[3])

    orig_roi = original_frame[y1:y2, x1:x2]
    result_roi = result_frame[y1:y2, x1:x2]

    if orig_roi.size == 0 or result_roi.size == 0:
        return 1.0

    orig_lab = cv2.cvtColor(orig_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    result_lab = cv2.cvtColor(result_roi, cv2.COLOR_BGR2LAB).astype(np.float32)

    diff = np.mean(np.abs(orig_lab - result_lab))
    return float(diff / 128.0)  # Normalize to roughly [0, 1]


def run_pipeline(name, frame, source_face, target_face, config):
    """Run a pipeline configuration and return metrics + result image."""
    # Apply config to globals
    for k, v in config.items():
        setattr(modules.globals, k, v)

    from modules.processors.frame.face_swapper import swap_face

    t0 = time.time()
    swapped = swap_face(source_face, target_face, frame.copy())
    swap_time = time.time() - t0

    # Apply enhancer if specified
    enhancer = config.get("_enhancer", None)
    enhance_time = 0.0
    if enhancer == "codeformer":
        import modules.processors.frame.face_enhancer_codeformer as cf_mod
        fidelity = config.get("_fidelity", 0.7)
        old_fidelity = cf_mod.FIDELITY_WEIGHT
        cf_mod.FIDELITY_WEIGHT = fidelity
        t1 = time.time()
        swapped = cf_mod.enhance_face(swapped, get_one_face(swapped) or target_face)
        enhance_time = time.time() - t1
        cf_mod.FIDELITY_WEIGHT = old_fidelity
    elif enhancer == "gpen256":
        try:
            from modules.processors.frame.face_enhancer_gpen256 import process_frame
            t1 = time.time()
            swapped = process_frame(None, swapped)
            enhance_time = time.time() - t1
        except Exception:
            pass
    elif enhancer == "gfpgan":
        try:
            from modules.processors.frame.face_enhancer import process_frame
            t1 = time.time()
            swapped = process_frame(None, swapped)
            enhance_time = time.time() - t1
        except Exception:
            pass

    # Compute metrics
    identity = compute_identity_score(source_face, swapped)
    detected_face = get_one_face(swapped)
    sharpness = compute_face_sharpness(swapped, detected_face)
    color_fid = compute_color_fidelity(frame, swapped, target_face)

    total_time = swap_time + enhance_time

    return {
        "name": name,
        "image": swapped,
        "identity": identity,
        "sharpness": sharpness,
        "color_fidelity": color_fid,
        "time": total_time,
    }


def run_hq_pipeline(name, frame, source_face, target_face, config):
    """Run the HIGH-QUALITY pipeline: paste_back=False + enhance aligned face + paste."""
    for k, v in config.items():
        if not k.startswith("_"):
            setattr(modules.globals, k, v)

    from modules.processors.frame.face_swapper import get_face_swapper
    from insightface.utils import face_align

    face_swapper = get_face_swapper()
    if face_swapper is None:
        return None

    t0 = time.time()

    # Step 1: Get raw 128x128 swapped face + affine matrix
    bgr_fake, M = face_swapper.get(frame.copy(), target_face, source_face, paste_back=False)
    # bgr_fake is 128x128 BGR uint8

    # Step 2: Get original aligned crop for color reference
    aimg, _ = face_align.norm_crop2(frame, target_face.kps, face_swapper.input_size[0])
    # aimg is 128x128 BGR uint8

    # Step 3: Color match swapped face to original skin tones (LAB space)
    # Use uint8 input for LAB: L=0-255, a=0-255, b=0-255
    fake_lab = cv2.cvtColor(bgr_fake, cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_lab = cv2.cvtColor(aimg, cv2.COLOR_BGR2LAB).astype(np.float32)

    for ch in range(3):
        f_mean = fake_lab[:, :, ch].mean()
        f_std = fake_lab[:, :, ch].std() + 1e-6
        o_mean = orig_lab[:, :, ch].mean()
        o_std = orig_lab[:, :, ch].std() + 1e-6
        # Partial transfer — blend toward original colors to keep identity
        blend = config.get("_color_blend", 0.4)
        target_mean = f_mean * (1 - blend) + o_mean * blend
        target_std = f_std * (1 - blend) + o_std * blend
        fake_lab[:, :, ch] = (fake_lab[:, :, ch] - f_mean) * (target_std / f_std) + target_mean

    color_matched = cv2.cvtColor(np.clip(fake_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Step 4: Upscale to 512x512 with Lanczos BEFORE enhancing
    upscaled = cv2.resize(color_matched, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    # Step 5: Apply enhancer on aligned face (not full frame)
    enhancer = config.get("_enhancer", None)
    fidelity = config.get("_fidelity", 0.7)
    enhanced = upscaled  # default if no enhancer

    if enhancer == "codeformer":
        model_path = os.path.join(PROJECT_ROOT, "models", "codeformer.onnx")
        if os.path.exists(model_path):
            from modules.processors.frame._onnx_enhancer import (
                create_onnx_session, preprocess_face, postprocess_face, THREAD_SEMAPHORE,
            )
            from modules.processors.frame.face_enhancer_codeformer import (
                get_enhancer, _run_codeformer,
            )
            session = get_enhancer()
            blob = preprocess_face(upscaled, 512)
            with THREAD_SEMAPHORE:
                output = _run_codeformer(session, blob, w=fidelity)
            enhanced = postprocess_face(output)

    # Step 6: Scale the affine matrix M to account for 128→512 upscale
    # M maps from frame → 128x128. We need frame → 512x512.
    scale_factor = 512.0 / 128.0
    M_scaled = M.copy()
    M_scaled[:, :2] *= scale_factor  # scale rotation/scale part
    M_scaled[:, 2] *= scale_factor   # scale translation

    # Step 7: Paste back with high-quality blending
    h, w_frame = frame.shape[:2]
    IM = cv2.invertAffineTransform(M_scaled)
    warped_face = cv2.warpAffine(
        enhanced, IM, (w_frame, h),
        flags=cv2.INTER_LANCZOS4,
        borderValue=(0, 0, 0),
    )

    # Create feathered mask at 512x512 then warp back
    mask_512 = np.ones((512, 512), dtype=np.float32)
    border = 32
    mask_512[:border, :] = np.linspace(0, 1, border)[:, np.newaxis]
    mask_512[-border:, :] = np.linspace(1, 0, border)[:, np.newaxis]
    mask_512[:, :border] = np.minimum(mask_512[:, :border], np.linspace(0, 1, border)[np.newaxis, :])
    mask_512[:, -border:] = np.minimum(mask_512[:, -border:], np.linspace(1, 0, border)[np.newaxis, :])

    warped_mask = cv2.warpAffine(
        mask_512, IM, (w_frame, h),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )

    mask_3ch = warped_mask[:, :, np.newaxis]
    result = (warped_face.astype(np.float32) * mask_3ch +
              frame.astype(np.float32) * (1.0 - mask_3ch))
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Step 8: Mouth mask (optional)
    if config.get("mouth_mask", False):
        from modules.processors.frame.mouth_handler import process_mouth_region
        try:
            result = process_mouth_region(
                swapped_frame=result,
                original_frame=frame,
                target_face=target_face,
                use_temporal_smoothing=False,
                use_poisson=True,
            )
        except Exception as e:
            print(f"Mouth mask failed in HQ pipeline: {e}")

    total_time = time.time() - t0

    # Compute metrics
    identity = compute_identity_score(source_face, result)
    detected_face = get_one_face(result)
    sharpness = compute_face_sharpness(result, detected_face)
    color_fid = compute_color_fidelity(frame, result, target_face)

    return {
        "name": name,
        "image": result,
        "identity": identity,
        "sharpness": sharpness,
        "color_fidelity": color_fid,
        "time": total_time,
    }


def main():
    print("=" * 70)
    print("EvilCam Likeness Evaluation")
    print("=" * 70)

    # Load assets
    src_img = cv2.imread(SOURCE_PATH)
    if src_img is None:
        print(f"FATAL: Source image not found: {SOURCE_PATH}")
        return
    source_face = get_one_face(src_img)
    if source_face is None:
        print("FATAL: No face in source image")
        return

    # Test multiple frames
    test_positions = [0.3, 0.5, 0.7]
    has_codeformer = os.path.exists(os.path.join(PROJECT_ROOT, "models", "codeformer.onnx"))

    # Pipeline configurations
    configs = [
        # Baseline configurations (current pipeline)
        ("A: Swap only (no enhance)", {
            "mouth_mask": False, "many_faces": False, "poisson_blend": False,
            "sharpness": 0.0, "opacity": 1.0, "swap_model": "inswapper",
            "_enhancer": None,
        }),
        ("B: Swap + mouth mask", {
            "mouth_mask": True, "many_faces": False, "poisson_blend": False,
            "sharpness": 0.0, "opacity": 1.0, "swap_model": "inswapper",
            "_enhancer": None,
        }),
        ("C: Swap + Poisson", {
            "mouth_mask": False, "many_faces": False, "poisson_blend": True,
            "sharpness": 0.0, "opacity": 1.0, "swap_model": "inswapper",
            "_enhancer": None,
        }),
        ("D: Swap + CodeFormer 0.7", {
            "mouth_mask": False, "many_faces": False, "poisson_blend": False,
            "sharpness": 0.0, "opacity": 1.0, "swap_model": "inswapper",
            "_enhancer": "codeformer", "_fidelity": 0.7,
        }),
        ("E: Swap + CodeFormer 0.5", {
            "mouth_mask": False, "many_faces": False, "poisson_blend": False,
            "sharpness": 0.0, "opacity": 1.0, "swap_model": "inswapper",
            "_enhancer": "codeformer", "_fidelity": 0.5,
        }),
        ("F: Swap + mouth + CodeFormer 0.7", {
            "mouth_mask": True, "many_faces": False, "poisson_blend": False,
            "sharpness": 0.0, "opacity": 1.0, "swap_model": "inswapper",
            "_enhancer": "codeformer",
        }),
    ]

    # High-quality pipeline configs
    hq_configs = [
        ("G: HQ swap + CF 0.7", {
            "mouth_mask": False, "swap_model": "inswapper",
            "_enhancer": "codeformer", "_fidelity": 0.7, "_color_blend": 0.4,
        }),
        ("H: HQ swap + CF 0.5", {
            "mouth_mask": False, "swap_model": "inswapper",
            "_enhancer": "codeformer", "_fidelity": 0.5, "_color_blend": 0.4,
        }),
        ("I: HQ swap + CF 0.3", {
            "mouth_mask": False, "swap_model": "inswapper",
            "_enhancer": "codeformer", "_fidelity": 0.3, "_color_blend": 0.4,
        }),
        ("J: HQ swap + CF 0.5 + mouth", {
            "mouth_mask": True, "swap_model": "inswapper",
            "_enhancer": "codeformer", "_fidelity": 0.5, "_color_blend": 0.4,
        }),
        ("K: HQ swap + CF 0.7 + color 0.6", {
            "mouth_mask": False, "swap_model": "inswapper",
            "_enhancer": "codeformer", "_fidelity": 0.7, "_color_blend": 0.6,
        }),
        ("L: HQ swap + no enhance (Lanczos only)", {
            "mouth_mask": False, "swap_model": "inswapper",
            "_enhancer": None, "_fidelity": 0.7, "_color_blend": 0.4,
        }),
    ]

    for pct in test_positions:
        frame = extract_frame(VIDEO_PATH, pct)
        if frame is None:
            print(f"Could not extract frame at {pct:.0%}")
            continue

        target_face = get_one_face(frame)
        if target_face is None:
            print(f"No face at {pct:.0%}")
            continue

        print(f"\n--- Frame at {pct:.0%} ---")
        print(f"{'Config':<40} {'Identity':>9} {'Sharp':>8} {'Color':>8} {'Time':>6}")
        print("-" * 75)

        all_results = []

        # Run standard configs
        for name, config in configs:
            if "_enhancer" in config and config["_enhancer"] == "codeformer" and not has_codeformer:
                print(f"{name:<40} {'SKIP':>9} (no codeformer model)")
                continue

            try:
                result = run_pipeline(name, frame, source_face, target_face, config)
                all_results.append(result)
                print(f"{result['name']:<40} {result['identity']:>8.4f} "
                      f"{result['sharpness']:>8.1f} {result['color_fidelity']:>7.4f} "
                      f"{result['time']:>5.2f}s")
            except Exception as e:
                print(f"{name:<40} ERROR: {e}")

        # Run HQ configs
        if has_codeformer:
            for name, config in hq_configs:
                try:
                    result = run_hq_pipeline(name, frame, source_face, target_face, config)
                    if result is None:
                        print(f"{name:<40} {'SKIP':>9}")
                        continue
                    all_results.append(result)
                    print(f"{result['name']:<40} {result['identity']:>8.4f} "
                          f"{result['sharpness']:>8.1f} {result['color_fidelity']:>7.4f} "
                          f"{result['time']:>5.2f}s")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"{name:<40} ERROR: {e}")

        # Save individual results
        for r in all_results:
            safe_name = r["name"].replace(":", "").replace(" ", "_").replace("+", "")
            fname = f"frame{int(pct*100)}_{safe_name}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, fname), r["image"])

        # Save comparison grid
        if all_results:
            save_comparison_grid(all_results, frame, pct)

        # Rank by identity score
        ranked = sorted(all_results, key=lambda x: x["identity"], reverse=True)
        print(f"\nRanked by identity (likeness to Jaimee):")
        for i, r in enumerate(ranked):
            marker = " <-- BEST" if i == 0 else ""
            print(f"  {i+1}. {r['name']:<40} identity={r['identity']:.4f} "
                  f"sharp={r['sharpness']:.0f}{marker}")

    # Save source face for reference
    cv2.imwrite(os.path.join(OUTPUT_DIR, "00_source_jaimee.jpg"), src_img)

    # Generate comparison videos for the top pipeline configs
    print("\n--- Generating Comparison Videos ---")
    generate_comparison_videos(source_face, has_codeformer)

    print(f"\nResults saved to {OUTPUT_DIR}")


def generate_comparison_videos(source_face, has_codeformer):
    """Generate full comparison videos at native framerate for each pipeline config."""
    from modules.processors.frame.face_swapper import get_face_swapper, swap_face
    from insightface.utils import face_align

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Source video: {total_frames} frames at {fps:.1f} fps, {w}x{h}")

    # Define video configs to render
    video_configs = [
        ("A_swap_only", {
            "mouth_mask": False, "many_faces": False, "poisson_blend": False,
            "sharpness": 0.0, "opacity": 1.0, "swap_model": "inswapper",
        }, "standard", None),
        ("B_swap_mouth", {
            "mouth_mask": True, "many_faces": False, "poisson_blend": False,
            "sharpness": 0.0, "opacity": 1.0, "swap_model": "inswapper",
        }, "standard", None),
        ("D_swap_CF07", {
            "mouth_mask": False, "many_faces": False, "poisson_blend": False,
            "sharpness": 0.0, "opacity": 1.0, "swap_model": "inswapper",
        }, "standard", "codeformer"),
        ("F_swap_mouth_CF07", {
            "mouth_mask": True, "many_faces": False, "poisson_blend": False,
            "sharpness": 0.0, "opacity": 1.0, "swap_model": "inswapper",
        }, "standard", "codeformer"),
    ]

    # Add HQ configs if CodeFormer available
    if has_codeformer:
        video_configs.extend([
            ("G_HQ_CF07", {
                "mouth_mask": False, "swap_model": "inswapper",
                "_enhancer": "codeformer", "_fidelity": 0.7, "_color_blend": 0.4,
            }, "hq", None),
            ("H_HQ_CF05", {
                "mouth_mask": False, "swap_model": "inswapper",
                "_enhancer": "codeformer", "_fidelity": 0.5, "_color_blend": 0.4,
            }, "hq", None),
            ("J_HQ_CF05_mouth", {
                "mouth_mask": True, "swap_model": "inswapper",
                "_enhancer": "codeformer", "_fidelity": 0.5, "_color_blend": 0.4,
            }, "hq", None),
        ])

    for vid_name, config, pipeline_type, enhancer_type in video_configs:
        if enhancer_type == "codeformer" and not has_codeformer:
            continue

        out_path = os.path.join(OUTPUT_DIR, f"video_{vid_name}.mp4")
        print(f"  Rendering {vid_name}...", end=" ", flush=True)

        # Apply config
        for k, v in config.items():
            if not k.startswith("_"):
                setattr(modules.globals, k, v)

        cap = cv2.VideoCapture(VIDEO_PATH)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        frame_scores = []
        t0 = time.time()

        swapper = get_face_swapper()

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            target_face = get_one_face(frame)
            if target_face is None:
                writer.write(frame)
                continue

            if pipeline_type == "hq":
                result = _process_frame_hq(
                    frame, source_face, target_face, swapper, config
                )
            else:
                result = swap_face(source_face, target_face, frame.copy())
                if enhancer_type == "codeformer":
                    from modules.processors.frame.face_enhancer_codeformer import enhance_face
                    result_face = get_one_face(result)
                    if result_face is not None:
                        result = enhance_face(result, result_face)

            writer.write(result)

            # Score every 10th frame
            if frame_idx % 10 == 0:
                score = compute_identity_score(source_face, result)
                frame_scores.append(score)

        cap.release()
        writer.release()

        elapsed = time.time() - t0
        avg_score = np.mean(frame_scores) if frame_scores else 0.0
        fps_actual = total_frames / elapsed if elapsed > 0 else 0
        print(f"done. {elapsed:.1f}s ({fps_actual:.1f} fps), avg identity={avg_score:.4f}")
        print(f"    -> {out_path}")


def _process_frame_hq(frame, source_face, target_face, swapper, config):
    """HQ pipeline for video: paste_back=False + enhance aligned face + paste."""
    from insightface.utils import face_align

    bgr_fake, M = swapper.get(frame.copy(), target_face, source_face, paste_back=False)

    # Color match in LAB
    aimg, _ = face_align.norm_crop2(frame, target_face.kps, swapper.input_size[0])
    fake_lab = cv2.cvtColor(bgr_fake, cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_lab = cv2.cvtColor(aimg, cv2.COLOR_BGR2LAB).astype(np.float32)

    blend = config.get("_color_blend", 0.4)
    for ch in range(3):
        f_mean = fake_lab[:, :, ch].mean()
        f_std = fake_lab[:, :, ch].std() + 1e-6
        o_mean = orig_lab[:, :, ch].mean()
        o_std = orig_lab[:, :, ch].std() + 1e-6
        target_mean = f_mean * (1 - blend) + o_mean * blend
        target_std = f_std * (1 - blend) + o_std * blend
        fake_lab[:, :, ch] = (fake_lab[:, :, ch] - f_mean) * (target_std / f_std) + target_mean

    color_matched = cv2.cvtColor(np.clip(fake_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Upscale + enhance
    upscaled = cv2.resize(color_matched, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    enhancer = config.get("_enhancer", None)
    fidelity = config.get("_fidelity", 0.7)
    enhanced = upscaled

    if enhancer == "codeformer":
        try:
            from modules.processors.frame._onnx_enhancer import preprocess_face, postprocess_face, THREAD_SEMAPHORE
            from modules.processors.frame.face_enhancer_codeformer import get_enhancer, _run_codeformer
            session = get_enhancer()
            blob = preprocess_face(upscaled, 512)
            with THREAD_SEMAPHORE:
                output = _run_codeformer(session, blob, w=fidelity)
            enhanced = postprocess_face(output)
        except Exception as e:
            print(f"CF enhance failed: {e}")

    # Paste back at 512 resolution
    scale_factor = 512.0 / 128.0
    M_scaled = M * scale_factor
    IM = cv2.invertAffineTransform(M_scaled)

    h, w_frame = frame.shape[:2]
    warped_face = cv2.warpAffine(
        enhanced, IM, (w_frame, h),
        flags=cv2.INTER_LANCZOS4, borderValue=(0, 0, 0),
    )

    mask_512 = np.ones((512, 512), dtype=np.float32)
    border = 32
    mask_512[:border, :] = np.linspace(0, 1, border)[:, np.newaxis]
    mask_512[-border:, :] = np.linspace(1, 0, border)[:, np.newaxis]
    mask_512[:, :border] = np.minimum(mask_512[:, :border], np.linspace(0, 1, border)[np.newaxis, :])
    mask_512[:, -border:] = np.minimum(mask_512[:, -border:], np.linspace(1, 0, border)[np.newaxis, :])

    warped_mask = cv2.warpAffine(mask_512, IM, (w_frame, h), flags=cv2.INTER_LINEAR, borderValue=0)

    # Mouth mask
    if config.get("mouth_mask", False):
        from modules.processors.frame.mouth_handler import process_mouth_region

    mask_3ch = warped_mask[:, :, np.newaxis]
    result = (warped_face.astype(np.float32) * mask_3ch +
              frame.astype(np.float32) * (1.0 - mask_3ch))
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Mouth mask post-composite
    if config.get("mouth_mask", False):
        try:
            from modules.processors.frame.mouth_handler import process_mouth_region
            result = process_mouth_region(
                swapped_frame=result,
                original_frame=frame,
                target_face=target_face,
                use_temporal_smoothing=True,
                use_poisson=True,
            )
        except Exception:
            pass

    return result


def save_comparison_grid(results, original_frame, pct):
    """Create a visual comparison grid of all pipeline results."""
    # Crop face regions for side-by-side comparison
    crops = []
    labels = []

    # Add original frame face crop first
    target_face = get_one_face(original_frame)
    if target_face is not None:
        bbox = target_face.bbox.astype(int)
        pad = 30
        x1 = max(0, bbox[0] - pad)
        y1 = max(0, bbox[1] - pad)
        x2 = min(original_frame.shape[1], bbox[2] + pad)
        y2 = min(original_frame.shape[0], bbox[3] + pad)
        crop = original_frame[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(cv2.resize(crop, (256, 256)))
            labels.append("ORIGINAL")

    for r in results:
        face = get_one_face(r["image"])
        if face is not None:
            bbox = face.bbox.astype(int)
            pad = 30
            x1 = max(0, bbox[0] - pad)
            y1 = max(0, bbox[1] - pad)
            x2 = min(r["image"].shape[1], bbox[2] + pad)
            y2 = min(r["image"].shape[0], bbox[3] + pad)
            crop = r["image"][y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(cv2.resize(crop, (256, 256)))
                short_name = r["name"][:20]
                labels.append(f"{short_name}\nID:{r['identity']:.3f} S:{r['sharpness']:.0f}")

    if not crops:
        return

    # Arrange in rows of 4
    cols = 4
    rows = (len(crops) + cols - 1) // cols
    cell_h = 256 + 50  # image + label space
    cell_w = 256
    grid = np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) * 40

    for i, (crop, label) in enumerate(zip(crops, labels)):
        r = i // cols
        c = i % cols
        y = r * cell_h
        x = c * cell_w
        grid[y:y + 256, x:x + 256] = crop

        # Add label
        for j, line in enumerate(label.split("\n")):
            cv2.putText(grid, line, (x + 5, y + 270 + j * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"grid_frame{int(pct * 100)}.jpg"), grid)
    print(f"  Grid saved: grid_frame{int(pct * 100)}.jpg")


if __name__ == "__main__":
    main()
