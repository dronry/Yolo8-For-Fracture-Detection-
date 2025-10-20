import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import random
import numpy as np


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.channel_attention(x)
        max_out = self.channel_attention(F.adaptive_max_pool2d(x, 1))
        ca = self.sigmoid(avg_out + max_out)
        x = x * ca

        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))
        x = x * sa

        return x

def inject_cbam_recursive(module, min_channels=128):
    """
    Recursively adds CBAM after convolutional layers
    with channels >= min_channels
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and child.out_channels >= min_channels:
            cbam = CBAMBlock(channels=child.out_channels)
            new_seq = nn.Sequential(child, cbam)
            setattr(module, name, new_seq)
        else:
            inject_cbam_recursive(child, min_channels)



def train_yolo_attention(data_yaml='fracatlas/data.yaml',
                          pretrained='yolov8s.pt',
                          epochs=100,
                          imgsz=640,
                          batch=16,
                          name='fracture_yolo8_attention',
                          device=0):

    set_seed()
    print("Loading YOLOv8 architecture only...")
    yolo = YOLO('yolov8s.yaml')  # Load structure only (no weights)

    model_container = yolo.model.model if hasattr(yolo.model, 'model') else yolo.model

    print("Injecting CBAM attention modules...")
    inject_cbam_recursive(model_container, min_channels=128)

    print("Loading pretrained weights (skip CBAM params)...")
    checkpoint = torch.load(pretrained, map_location='cpu')
    if 'model' in checkpoint:
        pretrained_dict = checkpoint['model'].float().state_dict()
    else:
        pretrained_dict = checkpoint

    model_dict = yolo.model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    yolo.model.load_state_dict(model_dict, strict=False)

    print("🚀 Starting training with CBAM-Enhanced YOLOv8...")
    yolo.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        workers=4,
        device=device
    )

    print("Training complete — model saved in `runs/detect/`.")



def eval_yolo_attention(weights='runs/detect/fracture_yolo8_attention/weights/best.pt',
                        data_yaml='fracatlas/data.yaml',
                        sample_image='xray_sample.png',
                        device=0):
    print("Loading trained model...")
    model = YOLO(weights)

    print(" Running inference on sample image...")
    results = model.predict(sample_image, device=device, save=True, conf=0.25)

    print("Evaluating on validation set...")
    metrics = model.val(data=data_yaml, device=device)

    print("Validation Results:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.precision:.4f}")
    print(f"Recall: {metrics.box.recall:.4f}")

    tp, fp, fn = metrics.box.tp, metrics.box.fp, metrics.box.fn
    total = tp + fp + fn
    acc = tp / total if total > 0 else 0.0
    print(f"Approx. Accuracy: {acc:.4f}")



if __name__ == "__main__":
    train_yolo_attention()
    eval_yolo_attention()




