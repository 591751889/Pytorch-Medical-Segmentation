import torch
import segmentation_models_pytorch as smp

# 使用改为224的输入尺寸来适配Swin Transformer
model = smp.Unet(
    encoder_name="tu-swin_tiny_patch4_window7_224",  # Swin-T encoder (timm‑unified → "tu-")
    encoder_weights="imagenet",                      # 或 None → 随机初始化
    in_channels=3,                                   # RGB 输入
    out_channels=1,                                       # 输出通道 = 类别数 (二分类=1)
)

x = torch.randn(4, 3, 224, 224)                      # (B, C, H, W) → (4, 3, 224, 224)
with torch.no_grad():
    y = model(x)

print("Input  shape:", x.shape)
print("Output shape:", y.shape)
