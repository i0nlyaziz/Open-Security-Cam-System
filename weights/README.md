# Model Weights

`yolo26n.pt` and `yolo26n.engine` are the official pretrained YOLO26 Nano model and its FP16 TensorRT engine used by this project — no custom training was done. They aren't committed directly to this repo (see `.gitignore`); instead, they're attached to this repo's Releases so anyone cloning it can download the exact same model files.

## Download

- **yolo26n.pt** — [Download from Releases](https://github.com/i0nlyaziz/Open-Security-Cam-System/releases/download/v1.0/yolo26n.pt)
- **yolo26n.engine** — [Download from Releases](https://github.com/i0nlyaziz/Open-Security-Cam-System/releases/download/v1.0/yolo26n.engine)

## Using the weights

1. Download the model file you want to use from the links above and place it in the `weights/` folder.
2. Configure `main.py` to load the corresponding model file — no export step is required.
3. Use `yolo26n.engine` (FP16) for optimized inference on compatible NVIDIA GPUs, or `yolo26n.pt` for the original PyTorch model. Note that TensorRT engines depend on hardware and software compatibility.