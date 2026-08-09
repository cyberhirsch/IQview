import sys
# stdin too, not just stdout: image paths arrive from C++ as UTF-8 and would
# otherwise be decoded with the locale codepage, corrupting any non-ASCII path.
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
print("STATUS: Starting segmentation worker...", flush=True)

import os
import argparse
import numpy as np
from PIL import Image, ImageFilter
import torch
print("STATUS: Loading segmentation libraries...", flush=True)
from transformers import Sam3Model, Sam3Processor
from huggingface_hub import login
from huggingface_hub import model_info as hf_model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
print("STATUS: Libraries loaded.", flush=True)

SAM3_MODEL_ID = "facebook/sam3"

# Fixed concept prompt for the subject cutout. SAM 3 is a text/box-prompted
# concept detector, not an automatic "segment everything" model — generic
# words like "object" or "thing" find nothing, but "background" reliably
# identifies the surrounding scene as a single concept (verified against
# both synthetic and photographic test images).
BACKGROUND_PROMPT = "background"

# Fail fast on stalled connections instead of hanging forever; the retry loop
# in _download_model picks up where the partial file left off.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")


def check_access(token=None):
    """Verify SAM 3 repo access without downloading weights. Returns True, 'GATED', or error str."""
    if token:
        login(token=token)
    try:
        hf_model_info(SAM3_MODEL_ID, token=token)
        return True
    except Exception as e:
        msg = str(e).lower()
        # Catch gated-repo signals regardless of exact exception class or HF version
        if ("gated" in msg or "403" in msg
                or ("access" in msg and "repo" in msg)
                or "accept" in msg):
            return "GATED"
        # Explicit class check as a secondary path (may not exist in older HF versions)
        try:
            if isinstance(e, (GatedRepoError, RepositoryNotFoundError)):
                return "GATED"
        except Exception:
            pass
        return str(e)


def _download_model(token=None):
    """Pre-download SAM 3 weights with progress reporting and automatic
    resume. snapshot_download() keeps partial files (*.incomplete) and
    continues them on retry, so a dropped connection never restarts from zero.
    A heartbeat thread reports cache growth as STATUS lines for the UI."""
    import threading
    import time
    from huggingface_hub import snapshot_download, constants

    # Total repo size from the metadata API, for a percentage display
    total = 0
    try:
        info = hf_model_info(SAM3_MODEL_ID, token=token, files_metadata=True)
        total = sum(s.size or 0 for s in (info.siblings or []))
    except Exception:
        pass

    cache_dir = os.path.join(constants.HF_HUB_CACHE, "models--facebook--sam3")

    def _dir_size(path):
        n = 0
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    n += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return n

    stop = threading.Event()

    def _heartbeat():
        while not stop.wait(timeout=3):
            done_mb = _dir_size(cache_dir) / 1024**2
            if total:
                pct = min(100.0, done_mb / (total / 1024**2) * 100)
                print(f"STATUS: Downloading SAM 3 — {done_mb:.0f} / {total / 1024**2:.0f} MB ({pct:.0f}%)",
                      flush=True)
            else:
                print(f"STATUS: Downloading SAM 3 — {done_mb:.0f} MB so far", flush=True)

    t = threading.Thread(target=_heartbeat, daemon=True)
    t.start()
    try:
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                snapshot_download(SAM3_MODEL_ID, token=token)
                break
            except Exception as e:
                if attempt == max_attempts:
                    raise
                print(f"STATUS: Download interrupted ({type(e).__name__}), "
                      f"resuming (attempt {attempt + 1}/{max_attempts})...", flush=True)
                time.sleep(3)
    finally:
        stop.set()
        t.join(timeout=1)


class IsolateWorker:
    def __init__(self, token=None):
        self.model  = None
        self.processor = None
        self.token  = token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        if self.token:
            login(token=self.token)
        print("STATUS: Checking SAM 3 weights (~3.2 GB on first run)...", flush=True)
        _download_model(self.token)
        print("STATUS: Loading SAM 3 model...", flush=True)
        # Load the concept-segmentation model directly rather than through
        # AutoModel/pipeline resolution, which for this repo resolves to the
        # unrelated Sam3VideoModel architecture (video tracking) and loads
        # with hundreds of randomly-initialized weights.
        self.model = Sam3Model.from_pretrained(SAM3_MODEL_ID, token=self.token or None).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=self.token or None)
        print("STATUS: SAM 3 ready.", flush=True)

    def remove_background(self, image_path, output_path):
        """Cut the subject out of the background using SAM 3's text-prompted
        concept segmentation, and save an RGBA PNG with the background made
        transparent."""
        image = Image.open(image_path).convert("RGB")
        W, H = image.size

        print("STATUS: Removing background with SAM 3...", flush=True)
        if self.model is None:
            self._load_model()

        inputs = self.processor(images=image, text=BACKGROUND_PROMPT, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.3, target_sizes=[(H, W)])[0]

        masks = results.get("masks", [])
        if len(masks) == 0:
            raise RuntimeError("Could not find a clear background region in this image.")

        # Background can be reported as several instances (e.g. sky + floor
        # counted separately) — union them into one background mask.
        background = np.zeros((H, W), dtype=bool)
        for m in masks:
            background |= np.asarray(m.cpu() if hasattr(m, "cpu") else m, dtype=bool)

        # Keep the subject (everything that ISN'T background); feather the
        # cutout edge a couple of pixels to avoid a jagged/aliased silhouette.
        subject_alpha = Image.fromarray((~background).astype(np.uint8) * 255, mode="L")
        subject_alpha = subject_alpha.filter(ImageFilter.GaussianBlur(2))

        rgba = image.convert("RGBA")
        rgba.putalpha(subject_alpha)
        rgba.save(output_path)
        print(f"OUTPUT: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    # Token arrives via HF_TOKEN env var (not on the command line, where it would be
    # visible in process listings). --token remains as a manual-testing fallback.
    if not args.token:
        args.token = os.environ.get("HF_TOKEN") or None

    # Verify access before loading the model so C++ can show the auth dialog.
    status = check_access(args.token)
    if status == "GATED":
        print("ACCESS_GATED", flush=True)
        return
    elif status is not True:
        print(f"FATAL: {status}", flush=True)
        return

    worker = IsolateWorker(token=args.token)

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            parts = line.strip().split("|")
            cmd = parts[0] if parts else ""

            if cmd == "REMOVE_BG" and len(parts) >= 3:
                image_path, output_path = parts[1], parts[2]
                worker.remove_background(image_path, output_path)

            else:
                if cmd:
                    print(f"ERROR: Unknown command: {cmd}", flush=True)

        except Exception as e:
            import traceback
            print(f"ERROR: {e}", flush=True)
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
