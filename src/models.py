"""Model definitions for brain tumor MRI classification.

Three architectures are provided:
    * CNN_TUMOR      - a lightweight custom CNN trained from scratch (input 256x256)
    * ResNet18 (TL)  - ImageNet-pretrained ResNet18 with a custom head (input 224x224)
    * MobileNetV2 (TL) - ImageNet-pretrained MobileNetV2 with a custom head (input 224x224)

Class indices follow torchvision ``ImageFolder`` ordering:
    0 -> "Brain Tumor", 1 -> "Healthy"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

CLASS_NAMES = ["Brain Tumor", "Healthy"]


def find_conv2d_out_shape(hin, win, conv, pool=2):
    """Return the (h, w) output size of ``conv`` followed by a ``pool`` x ``pool`` max-pool."""
    kernel_size, stride = conv.kernel_size, conv.stride
    padding, dilation = conv.padding, conv.dilation
    hout = np.floor((hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    wout = np.floor((win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    if pool:
        hout /= pool
        wout /= pool
    return int(hout), int(wout)


class CNN_TUMOR(nn.Module):
    """Custom 4-block CNN: Conv -> ReLU -> MaxPool (x4) -> FC(100) -> Dropout -> FC(2)."""

    def __init__(self, params=None):
        super().__init__()
        params = params or {
            "shape_in": (3, 256, 256),
            "initial_filters": 8,
            "num_fc1": 100,
            "dropout_rate": 0.25,
            "num_classes": 2,
        }
        cin, hin, win = params["shape_in"]
        init_f = params["initial_filters"]
        self.dropout_rate = params["dropout_rate"]

        self.conv1 = nn.Conv2d(cin, init_f, kernel_size=3)
        h, w = find_conv2d_out_shape(hin, win, self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2 * init_f, kernel_size=3)
        h, w = find_conv2d_out_shape(h, w, self.conv2)
        self.conv3 = nn.Conv2d(2 * init_f, 4 * init_f, kernel_size=3)
        h, w = find_conv2d_out_shape(h, w, self.conv3)
        self.conv4 = nn.Conv2d(4 * init_f, 8 * init_f, kernel_size=3)
        h, w = find_conv2d_out_shape(h, w, self.conv4)

        self.num_flatten = h * w * 8 * init_f
        self.fc1 = nn.Linear(self.num_flatten, params["num_fc1"])
        self.fc2 = nn.Linear(params["num_fc1"], params["num_classes"])

    def forward(self, x):
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
            x = F.max_pool2d(F.relu(conv(x)), 2, 2)
        x = x.view(-1, self.num_flatten)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def _head(in_features, num_classes=2):
    return nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
        nn.LogSoftmax(dim=1),
    )


def build_resnet18(pretrained=True, num_classes=2):
    """ResNet18 with frozen backbone except ``layer4`` (fine-tuned) and a new classifier head."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = _head(model.fc.in_features, num_classes)
    for p in model.layer4.parameters():
        p.requires_grad = True
    return model


def build_mobilenet_v2(pretrained=True, num_classes=2):
    """MobileNetV2 with a frozen feature extractor and a new classifier head."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    model.classifier = _head(model.last_channel, num_classes)
    return model


MODEL_REGISTRY = {
    "cnn": {"builder": lambda: CNN_TUMOR(), "input_size": 256},
    "resnet18": {"builder": lambda: build_resnet18(pretrained=False), "input_size": 224},
    "mobilenetv2": {"builder": lambda: build_mobilenet_v2(pretrained=False), "input_size": 224},
}


def load_model(arch, weights_path, device="cpu"):
    """Instantiate ``arch`` and load a saved ``state_dict`` from ``weights_path``."""
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {list(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[arch]["builder"]()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()
