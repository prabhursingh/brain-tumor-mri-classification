"""Run inference on one or more MRI images.

Examples
--------
python src/predict.py --arch cnn      --weights models/cnn_tumor.pt      path/to/scan.jpg
python src/predict.py --arch resnet18 --weights models/resnet18_brain.pt path/to/folder/
"""

import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import CLASS_NAMES, MODEL_REGISTRY, load_model  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def build_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def collect_images(paths):
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.lower().endswith(IMG_EXTS):
                    yield os.path.join(p, name)
        else:
            yield p


@torch.no_grad()
def predict(model, image_path, transform, device):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    probs = torch.exp(model(x))[0]  # models output log-probabilities
    idx = int(probs.argmax())
    return CLASS_NAMES[idx], float(probs[idx])


def main():
    parser = argparse.ArgumentParser(description="Brain tumor MRI classifier - inference")
    parser.add_argument("images", nargs="+", help="Image file(s) or folder(s)")
    parser.add_argument("--arch", choices=list(MODEL_REGISTRY), default="cnn")
    parser.add_argument("--weights", required=True, help="Path to a saved state_dict (.pt)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.arch, args.weights, device)
    transform = build_transform(MODEL_REGISTRY[args.arch]["input_size"])

    for path in collect_images(args.images):
        label, conf = predict(model, path, transform, device)
        print(f"{os.path.basename(path):40s} -> {label:12s} ({conf:.2%})")


if __name__ == "__main__":
    main()
