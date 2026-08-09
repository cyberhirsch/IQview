"""End-to-end smoke test for the iqView Python environment.

Runs against a freshly bootstrapped venv to answer one question: would a new
user on this platform actually be able to use the AI features? It checks the
things that have broken before and would otherwise only surface on a tester's
machine:

  * every dependency imports at all (a pinned version that no longer ships a
    wheel for this platform fails here rather than at first use -- this is how
    `onnxruntime-gpu` having no macOS wheel would have been caught);
  * the specific classes the features need still exist under the pinned
    versions (Flux2KleinPipeline for Creative Fill, Sam3Model for Isolate);
  * LaMa can actually run an inference, downloading its model if needed;
  * non-ASCII paths and alpha channels survive a real retouch round-trip.

Usage:  python smoke_test.py [--skip-inference]
Exit code is non-zero if anything fails.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# A directory name that is unrepresentable in most Windows ANSI codepages.
# Real users hit this simply by having a non-ASCII Windows username, since the
# worker's temp files live under their profile directory.
NON_ASCII_DIR = "iqview_тест_测试_ü"

failures = []


def check(name, fn):
    sys.stdout.write(f"  {name} ... ")
    sys.stdout.flush()
    try:
        detail = fn()
        print(f"OK{f' ({detail})' if detail else ''}")
        return True
    except Exception as exc:
        print("FAIL")
        traceback.print_exc()
        failures.append(f"{name}: {exc}")
        return False


def check_imports():
    print("\n[1/4] Dependency imports")

    def _onnxruntime():
        import onnxruntime as ort
        return f"{ort.__version__}, providers: {','.join(ort.get_available_providers())}"

    def _torch():
        import torch
        backend = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available() else "cpu")
        return f"{torch.__version__}, backend: {backend}"

    def _cv2():
        import cv2
        return cv2.__version__

    def _pil():
        import PIL
        return PIL.__version__

    def _numpy():
        import numpy
        return numpy.__version__

    check("onnxruntime", _onnxruntime)
    check("torch", _torch)
    check("torchvision", lambda: __import__("torchvision").__version__)
    check("opencv-python", _cv2)
    check("pillow", _pil)
    check("numpy", _numpy)


def check_feature_classes():
    """The pinned diffusers/transformers must still expose the exact classes the
    features import. A major upstream release that renames or moves these is the
    failure mode pinning is meant to prevent, and this is where it shows up."""
    print("\n[2/4] Feature entry points")

    def _flux():
        import diffusers
        from diffusers import Flux2KleinPipeline  # noqa: F401  (Creative Fill)
        return f"diffusers {diffusers.__version__}"

    def _sam3():
        import transformers
        from transformers import Sam3Model, Sam3Processor  # noqa: F401  (Isolate)
        return f"transformers {transformers.__version__}"

    check("diffusers.Flux2KleinPipeline", _flux)
    check("transformers.Sam3Model", _sam3)


def check_unicode_io():
    """Guards the imread_unicode/imwrite_unicode helpers in worker.py. Stock
    cv2.imread/imwrite silently fail on these paths on Windows."""
    print("\n[3/4] Non-ASCII path and alpha handling")

    import numpy as np
    import cv2
    sys.path.insert(0, SCRIPT_DIR)
    from worker import imread_unicode, imwrite_unicode

    d = os.path.join(tempfile.gettempdir(), NON_ASCII_DIR)
    os.makedirs(d, exist_ok=True)

    def _rgb_roundtrip():
        p = os.path.join(d, "rgb.png")
        img = np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
        if not imwrite_unicode(p, img):
            raise RuntimeError("imwrite_unicode returned False")
        back = imread_unicode(p, cv2.IMREAD_UNCHANGED)
        if back is None or back.shape != img.shape:
            raise RuntimeError(f"round-trip mismatch: {None if back is None else back.shape}")
        return None

    def _rgba_roundtrip():
        p = os.path.join(d, "rgba.png")
        img = np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
        rgba = np.dstack([img, np.full((32, 48), 128, np.uint8)])
        imwrite_unicode(p, rgba)
        back = imread_unicode(p, cv2.IMREAD_UNCHANGED)
        if back is None or back.shape[2] != 4:
            raise RuntimeError("alpha channel lost on round-trip")
        if int(back[0, 0, 3]) != 128:
            raise RuntimeError(f"alpha value corrupted: {int(back[0, 0, 3])}")
        return None

    check("BGR round-trip on non-ASCII path", _rgb_roundtrip)
    check("RGBA round-trip preserves alpha", _rgba_roundtrip)


def check_inference():
    """Drives worker.py over its real stdin/stdout protocol with an RGBA image
    on a non-ASCII path -- the exact combination that used to fail three
    different ways (path decoding, cv2 path handling, and alpha flattening)."""
    print("\n[4/4] LaMa inference round-trip")

    import numpy as np
    import cv2

    d = os.path.join(tempfile.gettempdir(), NON_ASCII_DIR)
    os.makedirs(d, exist_ok=True)
    img_p = os.path.join(d, "smoke_in.png")
    mask_p = os.path.join(d, "smoke_mask.bmp")
    out_p = os.path.join(d, "smoke_out.png")
    if os.path.exists(out_p):
        os.remove(out_p)

    h, w = 160, 240
    bgr = np.full((h, w, 3), 90, np.uint8)
    cv2.circle(bgr, (120, 80), 50, (30, 200, 30), -1)
    alpha = np.zeros((h, w), np.uint8)
    cv2.circle(alpha, (120, 80), 50, 255, -1)
    cv2.imencode(".png", np.dstack([bgr, alpha]))[1].tofile(img_p)

    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (120, 80), 12, 255, -1)
    cv2.imencode(".bmp", mask)[1].tofile(mask_p)

    def _run():
        proc = subprocess.Popen(
            [sys.executable, os.path.join(SCRIPT_DIR, "worker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, encoding="utf-8", cwd=SCRIPT_DIR)
        try:
            for line in proc.stdout:
                line = line.strip()
                if line == "READY":
                    break
                if line.startswith("FATAL"):
                    raise RuntimeError(line)
            else:
                raise RuntimeError("worker exited before becoming ready")

            proc.stdin.write(f"{img_p}|{mask_p}|{out_p}\n")
            proc.stdin.flush()

            for line in proc.stdout:
                line = line.strip()
                if line == "DONE":
                    break
                if line.startswith(("ERROR", "FATAL")):
                    raise RuntimeError(line)
            else:
                raise RuntimeError("worker exited before finishing the job")
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.terminate()
            proc.wait(timeout=10)

        if not os.path.exists(out_p):
            raise RuntimeError("worker reported DONE but wrote no output file")
        res = cv2.imdecode(np.fromfile(out_p, np.uint8), cv2.IMREAD_UNCHANGED)
        if res is None:
            raise RuntimeError("output file is not a readable image")
        if res.ndim != 3 or res.shape[2] != 4:
            raise RuntimeError(f"alpha channel lost through inpainting: shape {res.shape}")
        if int(res[0, 0, 3]) != 0 or int(res[80, 120, 3]) != 255:
            raise RuntimeError("alpha values altered by inpainting")
        return f"{res.shape[1]}x{res.shape[0]} RGBA"

    check("retouch a transparent PNG on a non-ASCII path", _run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-inference", action="store_true",
                        help="skip the LaMa run (avoids a ~200 MB model download)")
    args = parser.parse_args()

    print(f"iqView smoke test\npython: {sys.version.split()[0]} ({sys.executable})\n"
          f"platform: {sys.platform}")

    check_imports()
    check_feature_classes()
    check_unicode_io()
    if args.skip_inference:
        print("\n[4/4] LaMa inference round-trip ... SKIPPED")
    else:
        check_inference()

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
