import sys
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("No hf_hub_download")
    sys.exit(1)

try:
    path = hf_hub_download(repo_id="anyisalin/big-lama-onnx", filename="big-lama.onnx", local_dir="g:/Code/IQView/scripts")
    print("Found! " + path)
except Exception as e:
    print("Error:", e)
