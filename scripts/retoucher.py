import argparse
import os
import cv2
import numpy as np
import onnxruntime as ort
import sys
import time
from PIL import Image, ImageFilter

def log(msg, log_path):
    with open(log_path, "a") as f:
        f.write(msg + "\n")
    print(msg)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=None)
    return parser.parse_args()

def main():
    args = get_args()
    temp_dir = os.path.dirname(args.output)
    log_path = os.path.join(temp_dir, "iqview_retouch_log.txt")
    
    # Reset log
    with open(log_path, "w") as f:
        f.write("--- iqView Retouch Session Start ---\n")
    
    log(f"Input Image: {args.image}", log_path)
    log(f"Input Mask: {args.mask}", log_path)
    
    # Load image and mask
    img = cv2.imread(args.image)
    mask = cv2.imread(args.mask, 0)
    
    if img is None or mask is None:
        log(f"RET_ERR: Loading failed. Image: {img is not None}, Mask: {mask is not None}", log_path)
        return

    log(f"Image Shape: {img.shape}", log_path)
    log(f"Mask Shape: {mask.shape}", log_path)

    # Ensure mask is binary - more sensitive threshold
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
    # Find bounding box of the mask for ROI
    coords = cv2.findNonZero(mask)
    if coords is None:
        log("No mask pixels found! Skipping ROI.", log_path)
        cv2.imwrite(args.output, img)
        return

    log(f"Found {len(coords)} mask pixels.", log_path)
    x, y, w, h = cv2.boundingRect(coords)
    log(f"Bounding Box: x={x}, y={y}, w={w}, h={h}", log_path)
    
    # Add padding for context
    padding = max(w, h, 256) # Larger padding for better context
    h_orig, w_orig = img.shape[:2]
    
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_orig, x + w + padding)
    y2 = min(h_orig, y + h + padding)
    
    log(f"ROI Crop: {x1, y1} to {x2, y2}", log_path)
    
    # 1. Save ROI High-Res Crop for user
    roi_img = img[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]
    roi_h, roi_w = roi_img.shape[:2]
    cv2.imwrite(os.path.join(temp_dir, "iqview_retouch_roi_img.png"), roi_img)
    log("Saved iqview_retouch_roi_img.png", log_path)

    # Prepare model
    model_path = args.model
    if not model_path or not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), "big-lama.onnx")

    if not os.path.exists(model_path):
        log(f"Model missing at {model_path}. Attempting to download from HuggingFace...", log_path)
        try:
            from huggingface_hub import hf_hub_download
            import shutil
            hf_path = hf_hub_download(repo_id="anyisalin/big-lama-onnx", filename="onnx/model.onnx")
            shutil.copy(hf_path, model_path)
            log(f"Successfully downloaded to {model_path}", log_path)
        except Exception as e:
            log(f"RET_ERR: Failed to download model: {e}", log_path)
            return

    log(f"Using model: {model_path}", log_path)
    
    start_load = time.time()
    
    # Dynamically select only available providers to avoid warnings
    available = ort.get_available_providers()
    requested = ['DmlExecutionProvider', 'CUDAExecutionProvider']
    providers = [p for p in requested if p in available] + ['CPUExecutionProvider']
    
    # Session options to silence all debug/telemetry output
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3 # 3 = Error only
    sess_options.enable_mem_pattern = False # Often helps with DirectML stability
    
    session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
    load_time = time.time() - start_load
    
    log(f"Model loaded in {load_time:.3f}s. Active Providers: {session.get_providers()}", log_path)
    
    # Log model metadata for debugging
    for i, inp in enumerate(session.get_inputs()):
        log(f"Model Input {i}: name='{inp.name}', shape={inp.shape}, type={inp.type}", log_path)
    for i, out in enumerate(session.get_outputs()):
        log(f"Model Output {i}: name='{out.name}', shape={out.shape}, type={out.type}", log_path)
    
    # 2. Resize ROI to 512x512 and Save for user
    target_size = 512
    img_prep = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    img_prep = cv2.resize(img_prep, (target_size, target_size), interpolation=cv2.INTER_AREA)
    mask_prep = cv2.resize(roi_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(temp_dir, "iqview_retouch_roi_mask_512.png"), mask_prep)
    log("Saved iqview_retouch_roi_mask_512.png", log_path)
    
    # To Tensor
    t_img = img_prep.transpose(2, 0, 1).astype(np.float32) / 255.0
    t_mask = mask_prep[np.newaxis, ...].astype(np.float32) / 255.0
    
    # LaMa standard: hole should be 0 in input image
    t_img = t_img * (1.0 - t_mask)
    
    t_img = t_img[np.newaxis, ...]
    t_mask = t_mask[np.newaxis, ...]

    log(f"Tensor stats - Image: min={t_img.min():.3f}, max={t_img.max():.3f}", log_path)
    log(f"Tensor stats - Mask: min={t_mask.min():.3f}, max={t_mask.max():.3f}", log_path)

    inputs = {
        session.get_inputs()[0].name: t_img,
        session.get_inputs()[1].name: t_mask
    }
    
    # Inference with auto-fallback
    log("Running inference...", log_path)
    try:
        start_inf = time.time()
        output_raw = session.run(None, inputs)[0]
        inf_time = time.time() - start_inf
        log(f"Inference complete in {inf_time:.3f}s.", log_path)
    except Exception as e:
        log(f"RET_WARN: GPU inference failed, falling back to CPU. Error: {e}", log_path)
        # Force CPU session
        try:
            cpu_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'], sess_options=sess_options)
            start_inf = time.time()
            output_raw = cpu_session.run(None, inputs)[0]
            inf_time = time.time() - start_inf
            log(f"CPU Fallback Inference complete in {inf_time:.3f}s.", log_path)
        except Exception as e2:
            log(f"RET_ERR: Second-stage fallback failed: {e2}", log_path)
            sys.exit(1)
    
    # Handle range 0-1 vs 0-255
    res = output_raw[0]
    if output_raw.max() > 2.0:
        log("Detected 0-255 output range from model.", log_path)
    else:
        log("Detected 0-1 output range from model. Scaling by 255.", log_path)
        res = res * 255.0
    
    # 3. Save raw 512x512 Output for user
    out_img = res.transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(temp_dir, "iqview_retouch_roi_out_512.png"), out_img)
    log("Saved iqview_retouch_roi_out_512.png", log_path)
    
    # Resize output back to ROI size
    result_roi = cv2.resize(out_img, (roi_w, roi_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Blend and Paste
    mask_alpha = (roi_mask / 255.0)[:, :, np.newaxis].astype(np.float32)
    # Larger blur for smoother transitions and expansion
    mask_alpha = cv2.dilate(mask_alpha, np.ones((5, 5), np.uint8))
    mask_alpha = cv2.GaussianBlur(mask_alpha, (15, 15), 0)
    if mask_alpha.ndim == 2: mask_alpha = mask_alpha[..., np.newaxis]
    if len(mask_alpha.shape) == 2: mask_alpha = mask_alpha[..., np.newaxis]

    blended_roi = (result_roi.astype(np.float32) * mask_alpha + roi_img.astype(np.float32) * (1 - mask_alpha)).astype(np.uint8)
    
    final_img = img.copy()
    final_img[y1:y2, x1:x2] = blended_roi
    
    cv2.imwrite(args.output, final_img)
    log("SUCCESS - final_img saved.", log_path)

if __name__ == "__main__":
    main()
