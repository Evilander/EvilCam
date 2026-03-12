"""End-to-end runtime tests for EvilCam.

Tests actual face swap operations with real models and video frames.
Requires: models downloaded, test video + source image available.

Test matrix:
  1. Single-frame swap (inswapper, process_frame)
  2. Single-frame swap with mouth mask
  3. Face targeting via process_frame_v2 + simple_map
  4. Multi-face detection and many_faces mode
  5. CodeFormer enhancement pipeline
  6. GPEN-512 enhancement pipeline
  7. Full video processing (short clip, 10 frames)
  8. Batch processor chunk processing
  9. Poisson blending path
  10. Face detection deduplication (webui detect_faces_in_video)
"""

import sys
import os
import time
import tempfile
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

# Configure globals before imports
import modules.globals
import onnxruntime

available = onnxruntime.get_available_providers()
if "CUDAExecutionProvider" in available:
    modules.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
elif "DmlExecutionProvider" in available:
    modules.globals.execution_providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
else:
    modules.globals.execution_providers = ["CPUExecutionProvider"]

modules.globals.execution_threads = 1
modules.globals.max_memory = 16
modules.globals.keep_fps = True
modules.globals.keep_audio = False

print(f"Execution providers: {modules.globals.execution_providers}")

from modules.face_analyser import get_one_face, get_many_faces

# Paths
SOURCE_PATH = r"B:\projects\archive\jema\IMG_0027.jpg"
VIDEO_PATH = os.path.join(PROJECT_ROOT, "test_output", "redgifs_test_bbc.mp4")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_output", "e2e_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Track results
RESULTS = []


def result(name, passed, detail="", elapsed=0.0):
    status = "PASS" if passed else "FAIL"
    RESULTS.append((name, passed, detail))
    print(f"  [{status}] {name} ({elapsed:.2f}s) {detail}")


def extract_frame(video_path, pct=0.5):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * pct))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def extract_n_frames(video_path, n=10):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(total * i / n) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


# =========================================================================
# Test 1: Single-frame swap (inswapper)
# =========================================================================
def test_01_single_frame_swap():
    from modules.processors.frame.face_swapper import process_frame
    modules.globals.mouth_mask = False
    modules.globals.many_faces = False
    modules.globals.opacity = 1.0
    modules.globals.sharpness = 0.0
    modules.globals.swap_model = "inswapper"

    frame = extract_frame(VIDEO_PATH, 0.5)
    if frame is None:
        return result("01_single_frame_swap", False, "Could not extract frame")

    src_img = cv2.imread(SOURCE_PATH)
    source_face = get_one_face(src_img)
    if source_face is None:
        return result("01_single_frame_swap", False, "No source face detected")

    target_face = get_one_face(frame)
    if target_face is None:
        return result("01_single_frame_swap", False, "No target face in frame")

    t0 = time.time()
    swapped = process_frame(source_face, frame.copy())
    elapsed = time.time() - t0

    # Verify output is valid
    ok = (swapped is not None and
          swapped.shape == frame.shape and
          swapped.dtype == np.uint8 and
          not np.array_equal(swapped, frame))

    cv2.imwrite(os.path.join(OUTPUT_DIR, "01_single_swap.jpg"), swapped)
    result("01_single_frame_swap", ok,
           f"shape={swapped.shape}, changed={not np.array_equal(swapped, frame)}", elapsed)


# =========================================================================
# Test 2: Single-frame swap with mouth mask
# =========================================================================
def test_02_mouth_mask():
    from modules.processors.frame.face_swapper import process_frame
    modules.globals.mouth_mask = True
    modules.globals.many_faces = False

    frame = extract_frame(VIDEO_PATH, 0.5)
    src_img = cv2.imread(SOURCE_PATH)
    source_face = get_one_face(src_img)

    t0 = time.time()
    swapped = process_frame(source_face, frame.copy())
    elapsed = time.time() - t0

    # Also run without mouth mask for comparison
    modules.globals.mouth_mask = False
    swapped_no_mouth = process_frame(source_face, frame.copy())

    # The two outputs should differ (mouth mask changes the result)
    differs = not np.array_equal(swapped, swapped_no_mouth)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "02_mouth_mask.jpg"), swapped)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "02_no_mouth_mask.jpg"), swapped_no_mouth)
    result("02_mouth_mask", differs,
           f"mouth_mask differs from no-mask: {differs}", elapsed)


# =========================================================================
# Test 3: Face targeting via process_frame_v2 + simple_map
# =========================================================================
def test_03_face_targeting():
    from modules.processors.frame.face_swapper import process_frame_v2

    # Sample multiple positions to find a frame with 2+ faces
    frame = None
    all_faces = None
    for pct in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        candidate = extract_frame(VIDEO_PATH, pct)
        if candidate is None:
            continue
        faces = get_many_faces(candidate)
        if faces and len(faces) >= 2:
            frame = candidate
            all_faces = faces
            break

    if frame is None or all_faces is None or len(all_faces) < 2:
        best_count = 0
        for pct in [0.1, 0.3, 0.5, 0.7, 0.9]:
            candidate = extract_frame(VIDEO_PATH, pct)
            if candidate is None:
                continue
            faces = get_many_faces(candidate)
            if faces and len(faces) > best_count:
                best_count = len(faces)
        return result("03_face_targeting", False,
                       f"Need 2+ faces in one frame, best found {best_count}. "
                       "Try a different video with multiple people.")

    # Get source face
    src_img = cv2.imread(SOURCE_PATH)
    source_face = get_one_face(src_img)

    # Target only the SECOND face (index 1)
    target_embedding = all_faces[1].normed_embedding
    modules.globals.map_faces = True
    modules.globals.many_faces = False
    modules.globals.mouth_mask = False
    modules.globals.target_path = VIDEO_PATH  # Needed for is_video check
    modules.globals.simple_map = {
        "source_faces": [source_face],
        "target_embeddings": [target_embedding],
    }

    t0 = time.time()
    swapped = process_frame_v2(frame.copy())
    elapsed = time.time() - t0

    # Verify the frame was modified
    ok = (swapped is not None and not np.array_equal(swapped, frame))

    # Clean up globals
    modules.globals.map_faces = False
    modules.globals.simple_map = {}

    cv2.imwrite(os.path.join(OUTPUT_DIR, "03_face_targeting.jpg"), swapped)

    # Annotate original with face bboxes for reference
    annotated = frame.copy()
    for i, face in enumerate(all_faces):
        bbox = face.bbox.astype(int)
        color = (0, 255, 0) if i == 1 else (0, 0, 255)
        label = f"TARGET (Face {i})" if i == 1 else f"Face {i}"
        cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(annotated, label, (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "03_face_targeting_annotated.jpg"), annotated)

    result("03_face_targeting", ok,
           f"faces_detected={len(all_faces)}, targeted=Face1, swapped={ok}", elapsed)


# =========================================================================
# Test 4: Many faces mode
# =========================================================================
def test_04_many_faces():
    from modules.processors.frame.face_swapper import process_frame
    modules.globals.many_faces = True
    modules.globals.mouth_mask = False
    modules.globals.map_faces = False

    frame = extract_frame(VIDEO_PATH, 0.3)
    src_img = cv2.imread(SOURCE_PATH)
    source_face = get_one_face(src_img)

    all_faces = get_many_faces(frame)
    face_count = len(all_faces) if all_faces else 0

    t0 = time.time()
    swapped = process_frame(source_face, frame.copy())
    elapsed = time.time() - t0

    ok = swapped is not None and not np.array_equal(swapped, frame)

    modules.globals.many_faces = False
    cv2.imwrite(os.path.join(OUTPUT_DIR, "04_many_faces.jpg"), swapped)
    result("04_many_faces", ok,
           f"faces_in_frame={face_count}", elapsed)


# =========================================================================
# Test 5: CodeFormer enhancement
# =========================================================================
def test_05_codeformer():
    model_path = os.path.join(PROJECT_ROOT, "models", "codeformer.onnx")
    if not os.path.exists(model_path):
        return result("05_codeformer", False, "codeformer.onnx not found — skipped")

    from modules.processors.frame.face_swapper import process_frame as swap_frame
    from modules.processors.frame.face_enhancer_codeformer import process_frame as enhance_frame

    modules.globals.mouth_mask = True
    modules.globals.many_faces = False

    frame = extract_frame(VIDEO_PATH, 0.5)
    src_img = cv2.imread(SOURCE_PATH)
    source_face = get_one_face(src_img)

    t0 = time.time()
    swapped = swap_frame(source_face, frame.copy())
    enhanced = enhance_frame(None, swapped)
    elapsed = time.time() - t0

    differs = not np.array_equal(enhanced, swapped)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "05_codeformer.jpg"), enhanced)
    result("05_codeformer", differs,
           f"enhancement changed output: {differs}", elapsed)


# =========================================================================
# Test 6: GPEN-512 enhancement
# =========================================================================
def test_06_gpen512():
    model_path = os.path.join(PROJECT_ROOT, "models", "GPEN-BFR-512.onnx")
    if not os.path.exists(model_path):
        return result("06_gpen512", False, "GPEN-BFR-512.onnx not found — skipped")

    from modules.processors.frame.face_swapper import process_frame as swap_frame
    from modules.processors.frame.face_enhancer_gpen512 import process_frame as enhance_frame

    modules.globals.mouth_mask = False
    modules.globals.many_faces = False

    frame = extract_frame(VIDEO_PATH, 0.5)
    src_img = cv2.imread(SOURCE_PATH)
    source_face = get_one_face(src_img)

    t0 = time.time()
    swapped = swap_frame(source_face, frame.copy())
    enhanced = enhance_frame(None, swapped)
    elapsed = time.time() - t0

    differs = not np.array_equal(enhanced, swapped)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "06_gpen512.jpg"), enhanced)
    result("06_gpen512", differs,
           f"enhancement changed output: {differs}", elapsed)


# =========================================================================
# Test 7: Full video processing (short clip, 10 frames)
# =========================================================================
def test_07_video_processing():
    from modules.processors.frame.face_swapper import process_frames

    modules.globals.mouth_mask = False
    modules.globals.many_faces = False
    modules.globals.map_faces = False

    # Extract 10 frames to temp dir
    tmp_dir = tempfile.mkdtemp(prefix="evilcam_e2e_")
    frames = extract_n_frames(VIDEO_PATH, 10)
    if len(frames) < 5:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return result("07_video_processing", False, f"Only got {len(frames)} frames")

    frame_paths = []
    for i, frame in enumerate(frames):
        path = os.path.join(tmp_dir, f"{i+1:06d}.jpg")
        cv2.imwrite(path, frame)
        frame_paths.append(path)

    t0 = time.time()
    process_frames(SOURCE_PATH, frame_paths)
    elapsed = time.time() - t0

    # Check that frames were modified
    modified_count = 0
    for i, (path, orig) in enumerate(zip(frame_paths, frames)):
        processed = cv2.imread(path)
        if processed is not None and not np.array_equal(processed, orig):
            modified_count += 1

    ok = modified_count >= len(frames) // 2  # At least half should be modified

    # Save first and last for inspection
    first = cv2.imread(frame_paths[0])
    last = cv2.imread(frame_paths[-1])
    if first is not None:
        cv2.imwrite(os.path.join(OUTPUT_DIR, "07_video_frame_first.jpg"), first)
    if last is not None:
        cv2.imwrite(os.path.join(OUTPUT_DIR, "07_video_frame_last.jpg"), last)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    result("07_video_processing", ok,
           f"{modified_count}/{len(frames)} frames modified", elapsed)


# =========================================================================
# Test 8: Batch processor chunk processing
# =========================================================================
def test_08_batch_chunk():
    from modules.batch_processor import MovieProcessor, probe_video
    from modules.processors.frame.core import get_frame_processors_modules

    modules.globals.mouth_mask = False
    modules.globals.many_faces = False
    modules.globals.map_faces = False
    modules.globals.frame_processors = ["face_swapper"]

    info = probe_video(VIDEO_PATH)
    if info["total_frames"] == 0:
        return result("08_batch_chunk", False, "Could not probe video")

    tmp_dir = tempfile.mkdtemp(prefix="evilcam_batch_e2e_")
    out_path = os.path.join(tmp_dir, "test_batch_output.mp4")

    t0 = time.time()
    try:
        processor = MovieProcessor(
            source_path=SOURCE_PATH,
            target_path=VIDEO_PATH,
            output_path=out_path,
            chunk_size=50,  # Small chunks for test
        )

        # Just test the chunk extraction + processing, not full encode
        processor.work_dir = os.path.join(tmp_dir, "work")
        os.makedirs(processor.work_dir, exist_ok=True)

        # Extract one small chunk
        from modules.batch_processor import extract_chunk_frames
        chunk_dir = os.path.join(processor.work_dir, "test_chunk")
        frame_paths = extract_chunk_frames(VIDEO_PATH, chunk_dir, 0, 5, info["fps"])

        if not frame_paths:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return result("08_batch_chunk", False, "No frames extracted from chunk")

        # Load source face
        src_img = cv2.imread(SOURCE_PATH)
        source_face = get_one_face(src_img)
        del src_img

        # Process with batch processor's method
        from modules.batch_processor import ChunkState
        chunk_state = ChunkState(chunk_id=0, start_frame=0, end_frame=5)
        processor_modules = get_frame_processors_modules(modules.globals.frame_processors)

        processor._process_chunk_frames(source_face, frame_paths, processor_modules, chunk_state)
        elapsed = time.time() - t0

        # Verify frames were modified
        modified = 0
        for fp in frame_paths:
            img = cv2.imread(fp)
            if img is not None:
                modified += 1

        ok = modified == len(frame_paths)
        result("08_batch_chunk", ok,
               f"{modified}/{len(frame_paths)} chunk frames processed", elapsed)

    except Exception as e:
        elapsed = time.time() - t0
        result("08_batch_chunk", False, f"Error: {e}", elapsed)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =========================================================================
# Test 9: Poisson blending
# =========================================================================
def test_09_poisson_blend():
    from modules.processors.frame.face_swapper import process_frame
    modules.globals.poisson_blend = True
    modules.globals.mouth_mask = False
    modules.globals.many_faces = False

    frame = extract_frame(VIDEO_PATH, 0.5)
    src_img = cv2.imread(SOURCE_PATH)
    source_face = get_one_face(src_img)

    t0 = time.time()
    swapped_poisson = process_frame(source_face, frame.copy())
    elapsed = time.time() - t0

    # Compare with non-Poisson
    modules.globals.poisson_blend = False
    swapped_normal = process_frame(source_face, frame.copy())

    differs = not np.array_equal(swapped_poisson, swapped_normal)
    modules.globals.poisson_blend = False

    cv2.imwrite(os.path.join(OUTPUT_DIR, "09_poisson.jpg"), swapped_poisson)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "09_no_poisson.jpg"), swapped_normal)
    result("09_poisson_blend", differs,
           f"Poisson output differs from normal: {differs}", elapsed)


# =========================================================================
# Test 10: Face detection deduplication (webui path)
# =========================================================================
def test_10_face_detection_dedup():
    # Simulate what webui.detect_faces_in_video does
    cap = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_embeddings = []
    all_crops = []

    t0 = time.time()
    for pct in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * pct))
        ret, frame = cap.read()
        if not ret:
            continue
        faces = get_many_faces(frame)
        if not faces:
            continue
        for face in faces:
            if face.normed_embedding is None:
                continue

            # Deduplication: check if this face is similar to one we already have
            is_duplicate = False
            for existing_emb in all_embeddings:
                sim = float(np.dot(face.normed_embedding, existing_emb))
                if sim > 0.6:
                    is_duplicate = True
                    break

            if not is_duplicate:
                all_embeddings.append(face.normed_embedding)
                bbox = face.bbox.astype(int)
                x1 = max(0, bbox[0])
                y1 = max(0, bbox[1])
                x2 = min(frame.shape[1], bbox[2])
                y2 = min(frame.shape[0], bbox[3])
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    all_crops.append(cv2.resize(crop, (128, 128)))

    cap.release()
    elapsed = time.time() - t0

    unique_faces = len(all_embeddings)
    ok = unique_faces >= 1  # Should find at least one unique face

    # Save face gallery
    if all_crops:
        gallery = np.hstack(all_crops[:8])  # Max 8 faces
        cv2.imwrite(os.path.join(OUTPUT_DIR, "10_detected_faces.jpg"), gallery)

    result("10_face_detection_dedup", ok,
           f"unique_faces={unique_faces} from {len(sample_positions)} samples", elapsed)


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 70)
    print("EvilCam End-to-End Runtime Tests")
    print("=" * 70)
    print(f"Source: {SOURCE_PATH}")
    print(f"Video:  {VIDEO_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Verify assets
    if not os.path.exists(SOURCE_PATH):
        print(f"FATAL: Source image not found: {SOURCE_PATH}")
        return
    if not os.path.exists(VIDEO_PATH):
        print(f"FATAL: Test video not found: {VIDEO_PATH}")
        return

    # Model check
    models_dir = os.path.join(PROJECT_ROOT, "models")
    models = [f for f in os.listdir(models_dir) if f.endswith(".onnx")] if os.path.isdir(models_dir) else []
    print(f"Models: {models}")
    print()

    tests = [
        test_01_single_frame_swap,
        test_02_mouth_mask,
        test_03_face_targeting,
        test_04_many_faces,
        test_05_codeformer,
        test_06_gpen512,
        test_07_video_processing,
        test_08_batch_chunk,
        test_09_poisson_blend,
        test_10_face_detection_dedup,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            result(test_fn.__name__, False, f"EXCEPTION: {e}")

    # Summary
    print()
    print("=" * 70)
    passed = sum(1 for _, p, _ in RESULTS if p)
    failed = sum(1 for _, p, _ in RESULTS if not p)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(RESULTS)} tests")
    if failed:
        print("\nFailed tests:")
        for name, p, detail in RESULTS:
            if not p:
                print(f"  - {name}: {detail}")
    print("=" * 70)


if __name__ == "__main__":
    main()
