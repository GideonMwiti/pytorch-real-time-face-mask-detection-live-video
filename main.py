"""
Project 68: Face Mask Detection on Live Video Feed
Detects and classifies human faces as 'Mask' or 'No Mask' in real-time video
streams using an edge-optimized PyTorch 2D-CNN (MaskNet), confidence thresholding,
and live video stream bounding-box compliance annotation.
"""

import os
import sys
import time
import json
import random
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------
# Global Configuration & Hardware Setup
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["No_Mask", "Mask"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
IMG_SIZE = 64

# ---------------------------------------------------------
# Parametric Face & Mask Generator
# ---------------------------------------------------------
def synthesize_face_mask_sample(has_mask=True, mask_type=0, size=64):
    """
    Renders realistic 64x64 RGB face crop with or without protective face mask:
    - Mask Type 0: Surgical Light Blue mask with pleats and white trim.
    - Mask Type 1: N95 Particulate Respirator with convex cone and metal clip.
    - Mask Type 2: Black Fabric / Cloth mask with contoured chin cup.
    """
    img = np.zeros((size, size, 3), dtype=np.float32)
    y, x = np.ogrid[:size, :size]

    # 1. Base Head & Skin
    cy, cx = 32.0 + np.random.uniform(-0.5, 0.5), 32.0 + np.random.uniform(-0.5, 0.5)
    head_mask = (((y - cy) / 25.0) ** 2 + ((x - cx) / 18.0) ** 2) <= 1.0

    skin_palette = [
        [0.85, 0.70, 0.60],  # Light
        [0.78, 0.62, 0.50],  # Tan
        [0.65, 0.48, 0.38],  # Olive
        [0.45, 0.32, 0.24]   # Dark Brown
    ]
    base_skin = np.array(random.choice(skin_palette), dtype=np.float32)
    dist = np.sqrt((y - cy)**2 + (x - cx)**2) / 30.0
    shading = np.clip(1.0 - 0.25 * dist, 0.55, 1.0)

    for c in range(3):
        img[head_mask, c] = base_skin[c] * shading[head_mask]

    # 2. Eyes & Eyebrows
    eye_y = cy - 6.5
    for ex in [cx - 10.5, cx + 10.5]:
        eye_mask = (((y - eye_y) / 2.8) ** 2 + ((x - ex) / 3.6) ** 2) <= 1.0
        img[eye_mask] = [0.95, 0.95, 0.95]
        pupil_mask = (((y - eye_y) / 1.6) ** 2 + ((x - ex) / 1.8) ** 2) <= 1.0
        img[pupil_mask] = [0.15, 0.12, 0.10]

    brow_y = eye_y - 4.5
    for i in range(-7, 8):
        for bx in [cx - 10.5 + i, cx + 10.5 + i]:
            by = int(brow_y - 0.03 * (i ** 2))
            if 0 <= by < size and 0 <= int(bx) < size:
                img[by:by + 2, int(bx)] = [0.20, 0.15, 0.10]

    # 3. Nose Bridge
    img[int(cy - 3):int(cy + 6), int(cx) - 1:int(cx) + 2] = base_skin * 0.80

    if not has_mask:
        # 4A. Unmasked Features: Exposed Mouth, Lips & Nostrils
        # Nostrils
        img[int(cy + 6), int(cx - 3)] = [0.20, 0.15, 0.12]
        img[int(cy + 6), int(cx + 3)] = [0.20, 0.15, 0.12]

        # Lips
        mouth_y = cy + 13.0
        lip_color = np.clip(base_skin * [1.1, 0.65, 0.70], 0.0, 1.0)
        mouth_mask = (((y - mouth_y) / 2.6) ** 2 + ((x - cx) / 7.5) ** 2) <= 1.0
        img[mouth_mask] = lip_color
        img[int(mouth_y), int(cx - 6):int(cx + 7)] = [0.25, 0.12, 0.12]

    else:
        # 4B. Mask Features: Complete Coverage from Nose-Bridge to Chin
        mask_top_y = cy + 1.0   # Covers middle of nose
        mask_bot_y = cy + 24.0  # Wraps under chin
        mask_half_w = 17.0

        # Mask boundary contour
        mask_region = (y >= mask_top_y) & (y <= mask_bot_y) & (np.abs(x - cx) <= mask_half_w) & head_mask

        if mask_type == 0:
            # Surgical Blue Mask: [0.35, 0.65, 0.85] with horizontal pleats & white trim
            mask_base_color = np.array([0.35, 0.68, 0.88], dtype=np.float32)
            for c in range(3):
                img[mask_region, c] = mask_base_color[c]

            # Horizontal Pleats
            for py in [int(cy + 7), int(cy + 13), int(cy + 19)]:
                if py < size:
                    pleat_line = (y == py) & mask_region
                    for c in range(3):
                        img[pleat_line, c] *= 0.70

            # White Top Seam Trim
            top_trim = (y >= mask_top_y) & (y <= mask_top_y + 1.5) & mask_region
            img[top_trim] = [0.95, 0.95, 0.95]

        elif mask_type == 1:
            # N95 White Respirator: [0.92, 0.92, 0.90] with metal nose clip
            mask_base_color = np.array([0.92, 0.92, 0.90], dtype=np.float32)
            # Convex cone shading
            cone_dist = np.sqrt(((y - (cy + 12)) / 12.0)**2 + ((x - cx) / 14.0)**2)
            cone_shading = np.clip(1.05 - 0.25 * cone_dist, 0.65, 1.1)

            for c in range(3):
                img[mask_region, c] = mask_base_color[c] * cone_shading[mask_region]

            # Metallic nose clip
            clip_mask = (y >= mask_top_y) & (y <= mask_top_y + 2.0) & (np.abs(x - cx) <= 6.0) & mask_region
            img[clip_mask] = [0.75, 0.75, 0.78]

            # Vertical center seam
            seam_mask = (np.abs(x - cx) <= 0.8) & mask_region
            for c in range(3):
                img[seam_mask, c] *= 0.75

        else:
            # Black Fabric / Cloth Mask: [0.15, 0.15, 0.16]
            mask_base_color = np.array([0.15, 0.15, 0.16], dtype=np.float32)
            weave = 1.0 + 0.08 * np.sin(x * 3.0) * np.cos(y * 3.0)
            for c in range(3):
                img[mask_region, c] = np.clip(mask_base_color[c] * weave[mask_region], 0.0, 1.0)

        # White/Elastic Ear Loops
        for ly in range(int(mask_top_y + 4), int(mask_bot_y - 4), 2):
            lx_left = int(cx - mask_half_w - 1)
            lx_right = int(cx + mask_half_w + 1)
            if 0 <= ly < size:
                if 0 <= lx_left < size: img[ly, lx_left] = [0.90, 0.90, 0.90]
                if 0 <= lx_right < size: img[ly, lx_right] = [0.90, 0.90, 0.90]

    # Gaussian smoothing & sensor noise
    for c in range(3):
        img[:, :, c] = scipy.ndimage.gaussian_filter(img[:, :, c], sigma=0.55)
    noise = np.random.normal(0, 0.02, (size, size, 3))
    return np.clip(img + noise, 0.0, 1.0).astype(np.float32)

# ---------------------------------------------------------
# Dataset & Augmentation Pipeline
# ---------------------------------------------------------
class FaceMaskDataset(Dataset):
    def __init__(self, samples, is_train=True):
        self.samples = samples
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (3, 64, 64)

        if self.is_train:
            # 1. Random Horizontal Flip (p=0.5)
            if random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, dims=[2])

            # 2. Random Affine Rotation & Jitter
            angle = random.uniform(-10.0, 10.0)
            rad = np.deg2rad(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            tx, ty = random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)
            theta = torch.tensor([[cos_a, -sin_a, tx],
                                  [sin_a,  cos_a, ty]], dtype=torch.float32).unsqueeze(0)
            grid = F.affine_grid(theta, img_tensor.unsqueeze(0).size(), align_corners=False)
            img_tensor = F.grid_sample(img_tensor.unsqueeze(0), grid, mode='bilinear',
                                       padding_mode='border', align_corners=False).squeeze(0)

            # 3. Brightness & Contrast Jitter
            b_factor = random.uniform(0.88, 1.12)
            c_factor = random.uniform(0.88, 1.12)
            img_tensor = torch.clamp((img_tensor - 0.5) * c_factor + 0.5 * b_factor, 0.0, 1.0)

        # Normalize to standard [-1, 1] range
        img_norm = (img_tensor - 0.5) / 0.5
        return img_norm, torch.tensor(label, dtype=torch.long)

# ---------------------------------------------------------
# Edge-Optimized Neural Network Architecture: MaskNet
# ---------------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   stride=1, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))

class MaskNet(nn.Module):
    """
    Lightweight, high-speed 2D Convolutional Neural Network designed
    specifically for real-time edge face mask detection on video streams.
    """
    def __init__(self, num_classes=2):
        super(MaskNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (32, 32, 32)
        )
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (64, 16, 16)
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (128, 8, 8)
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(128, 128),
            nn.AdaptiveAvgPool2d((1, 1))           # (128, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x)

# ---------------------------------------------------------
# Training & Evaluation Loops
# ---------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, (correct / total) * 100.0

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Prob of Mask
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (running_loss / total, (correct / total) * 100.0,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))

# ---------------------------------------------------------
# Master Execution Pipeline
# ---------------------------------------------------------
def main():
    print("=" * 80)
    print("  Project 68: Real-Time Face Mask Detection on Live Video Feed")
    print(f"  Target Classes: {CLASSES} (0=No_Mask, 1=Mask)")
    print(f"  Hardware Compute Device: {DEVICE}")
    print("=" * 80)

    # 1. Dataset Generation: 300 Masked + 300 Unmasked = 600 face crops
    samples_per_class = 300
    total_samples = samples_per_class * 2
    print(f"\n[1/6] Synthesizing {total_samples} face crops ({samples_per_class} Mask, {samples_per_class} No_Mask)...")

    raw_data = []
    prototypes = {"No_Mask": None, "Mask_Surgical": None, "Mask_N95": None, "Mask_Cloth": None}

    # Generate unmasked faces
    for i in range(samples_per_class):
        face = synthesize_face_mask_sample(has_mask=False, size=IMG_SIZE)
        raw_data.append((face, 0))
        if i == 0: prototypes["No_Mask"] = face

    # Generate masked faces with diverse mask types
    for i in range(samples_per_class):
        m_type = i % 3
        face = synthesize_face_mask_sample(has_mask=True, mask_type=m_type, size=IMG_SIZE)
        raw_data.append((face, 1))
        if m_type == 0 and prototypes["Mask_Surgical"] is None: prototypes["Mask_Surgical"] = face
        if m_type == 1 and prototypes["Mask_N95"] is None: prototypes["Mask_N95"] = face
        if m_type == 2 and prototypes["Mask_Cloth"] is None: prototypes["Mask_Cloth"] = face

    random.shuffle(raw_data)

    n_tr = int(0.70 * total_samples)
    n_va = int(0.15 * total_samples)
    train_data = raw_data[:n_tr]
    val_data = raw_data[n_tr:n_tr + n_va]
    test_data = raw_data[n_tr + n_va:]

    print(f"      Split breakdown: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    train_loader = DataLoader(FaceMaskDataset(train_data, is_train=True), batch_size=32, shuffle=True)
    val_loader = DataLoader(FaceMaskDataset(val_data, is_train=False), batch_size=32, shuffle=False)
    test_loader = DataLoader(FaceMaskDataset(test_data, is_train=False), batch_size=32, shuffle=False)

    # 2. Model Initialization & Complexity Profiling
    print("\n[2/6] Building Edge-Optimized MaskNet Architecture...")
    model = MaskNet(num_classes=2).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_kb = (total_params * 4) / 1024.0

    print(f"      Total Trainable Parameters: {total_params:,}")
    print(f"      Memory Footprint: {model_size_kb:.2f} KB (Ultra-Lightweight Edge Model)")

    # 3. Model Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    epochs = 20
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\n[3/6] Training MaskNet for {epochs} Epochs with Cosine Annealing...")
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    best_weights_path = os.path.join(os.path.dirname(__file__), "outputs", "masknet_best.pth")

    t_start = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        va_loss, va_acc, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(va_loss)
        val_accs.append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_weights_path)

        if ep % 4 == 0 or ep == epochs:
            print(f"      Epoch [{ep:02d}/{epochs:02d}] | Train Loss: {tr_loss:.4f} (Acc: {tr_acc:5.1f}%) | Val Loss: {va_loss:.4f} (Acc: {va_acc:5.1f}%) | LR: {scheduler.get_last_lr()[0]:.6f}")

    train_duration = time.time() - t_start
    print(f"      MaskNet training converged in {train_duration:.2f}s! Best Val Accuracy: {best_val_acc:.2f}%")

    # Load best model for evaluation
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))

    # 4. Rigorous Test Set Evaluation & Metrics
    print("\n[4/6] Evaluating on Unseen Test Dataset...")
    test_loss, test_acc, test_preds, test_targets, test_probs = evaluate(model, test_loader, criterion, DEVICE)

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(test_targets, test_preds):
        cm[t, p] += 1

    tp = cm[1, 1]  # True Masked
    fp = cm[0, 1]  # False Masked
    fn = cm[1, 0]  # False Unmasked
    tn = cm[0, 0]  # True Unmasked

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = (2 * prec * rec) / max(prec + rec, 1e-6)

    # ROC / AUC Calculation
    th_steps = np.linspace(0.0, 1.0, 100)
    tpr_list, fpr_list = [], []
    for th in th_steps:
        tp_i = sum(p >= th for p, t in zip(test_probs, test_targets) if t == 1)
        fn_i = sum(p < th for p, t in zip(test_probs, test_targets) if t == 1)
        fp_i = sum(p >= th for p, t in zip(test_probs, test_targets) if t == 0)
        tn_i = sum(p < th for p, t in zip(test_probs, test_targets) if t == 0)
        tpr_list.append(tp_i / max(tp_i + fn_i, 1))
        fpr_list.append(fp_i / max(fp_i + tn_i, 1))

    try:
        auc_score = float(np.trapezoid(tpr_list[::-1], fpr_list[::-1]))
    except AttributeError:
        auc_score = float(np.trapz(tpr_list[::-1], fpr_list[::-1]))
    auc_score = min(1.0, max(0.0, auc_score))

    # Latency benchmark
    sample_crop = ((torch.tensor(test_data[0][0], dtype=torch.float32).permute(2, 0, 1) - 0.5) / 0.5).unsqueeze(0).to(DEVICE)
    model.eval()
    latencies = []
    with torch.no_grad():
        for _ in range(100):
            t0 = time.perf_counter()
            _ = model(sample_crop)
            latencies.append((time.perf_counter() - t0) * 1000.0)
    mean_lat_ms = float(np.mean(latencies))

    print(f"      Test Accuracy : {test_acc:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"      ROC AUC Score : {auc_score:.4f}")
    print(f"      Crop Latency  : {mean_lat_ms:.2f} ms ({1000.0 / mean_lat_ms:.1f} FPS on Single CPU Core)")

    # 5. Live Video Feed Scene Simulation
    print("\n[5/6] Simulating Multi-Person Live Surveillance Camera Stream...")
    # Create simulated 640x360 security camera canvas
    cam_h, cam_w = 360, 640
    cam_canvas = np.full((cam_h, cam_w, 3), 0.12, dtype=np.float32)
    # Background office corridor gradient
    for y_idx in range(cam_h):
        cam_canvas[y_idx, :] += 0.08 * (y_idx / cam_h)

    # 4 Persons in the scene: 2 with masks, 2 without masks
    scene_persons = [
        {"bbox": [60, 70, 96, 96],   "face": synthesize_face_mask_sample(True, 0), "label": "Mask"},
        {"bbox": [210, 50, 104, 104], "face": synthesize_face_mask_sample(False),    "label": "No_Mask"},
        {"bbox": [370, 75, 96, 96],   "face": synthesize_face_mask_sample(True, 1), "label": "Mask"},
        {"bbox": [500, 60, 100, 100], "face": synthesize_face_mask_sample(False),    "label": "No_Mask"}
    ]

    detected_faces_summary = []
    for p in scene_persons:
        bx, by, bw, bh = p["bbox"]
        # Resize face into canvas
        face_resized = scipy.ndimage.zoom(p["face"], (bh / IMG_SIZE, bw / IMG_SIZE, 1), order=1)
        face_resized = np.clip(face_resized[:bh, :bw], 0.0, 1.0)
        cam_canvas[by:by + bh, bx:bx + bw] = face_resized

        # Run model inference on crop
        crop_t = ((torch.tensor(p["face"], dtype=torch.float32).permute(2, 0, 1) - 0.5) / 0.5).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out_logits = model(crop_t)
            prob_mask = torch.softmax(out_logits, dim=1)[0, 1].item()
            pred_class = "Mask" if prob_mask >= 0.50 else "No_Mask"
            conf = prob_mask if pred_class == "Mask" else (1.0 - prob_mask)

        detected_faces_summary.append({
            "bbox": p["bbox"],
            "pred": pred_class,
            "conf": conf,
            "is_compliant": (pred_class == "Mask")
        })

    # 6. Render Diagnostic Visual Dashboard
    print("\n[6/6] Rendering High-Resolution 6-Panel Diagnostic Dashboard...")
    fig = plt.figure(figsize=(19, 13))
    plt.subplots_adjust(hspace=0.38, wspace=0.25)

    # Panel 1: Face Crop Prototypes (Mask Variations vs Unmasked)
    ax1 = plt.subplot(2, 3, 1)
    proto_strip = np.hstack([prototypes["No_Mask"], prototypes["Mask_Surgical"],
                             prototypes["Mask_N95"], prototypes["Mask_Cloth"]])
    ax1.imshow(proto_strip)
    ax1.set_title("Dataset Prototypes: Unmasked vs Mask Types", fontsize=11, fontweight='bold')
    ax1.set_xticks([32, 96, 160, 224])
    ax1.set_xticklabels(["No Mask", "Surgical Blue", "N95 Respirator", "Black Cloth"], rotation=25, ha='right')
    ax1.set_yticks([])

    # Panel 2: Live Video Feed Multi-Person Detection Scene
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(cam_canvas)
    for det in detected_faces_summary:
        bx, by, bw, bh = det["bbox"]
        color = "#2ca02c" if det["is_compliant"] else "#d62728"
        rect = plt.Rectangle((bx, by), bw, bh, linewidth=2.5, edgecolor=color, facecolor='none')
        ax2.add_patch(rect)
        tag = f"{det['pred'].upper()}: {det['conf']*100:.1f}%"
        ax2.text(bx, by - 6, tag, fontsize=8, fontweight='bold', color='white',
                 bbox=dict(boxstyle="square,pad=0.2", facecolor=color, edgecolor='none'))

    compliant_count = sum(d["is_compliant"] for d in detected_faces_summary)
    hud_txt = f"SECURITY HUD | Total: {len(detected_faces_summary)} | Compliant: {compliant_count} | Violations: {len(detected_faces_summary) - compliant_count}"
    ax2.text(10, 25, hud_txt, fontsize=8, fontweight='bold', color='white',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.75))
    ax2.set_title("Live Surveillance Stream (Automated Mask Compliance)", fontsize=11, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Panel 3: Training & Validation Loss Curves
    ax3 = plt.subplot(2, 3, 3)
    ep_axis = range(1, epochs + 1)
    ax3.plot(ep_axis, train_losses, label="Train Loss", color="#1f77b4", linewidth=2.0)
    ax3.plot(ep_axis, val_losses, label="Val Loss", color="#ff7f0e", linestyle="--", linewidth=2.0)
    ax3.set_title("Cross-Entropy Loss Convergence", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(frameon=True)

    # Panel 4: Training & Validation Accuracy Dynamics
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(ep_axis, train_accs, label="Train Accuracy", color="#2ca02c", linewidth=2.0)
    ax4.plot(ep_axis, val_accs, label="Val Accuracy", color="#d62728", linestyle="--", linewidth=2.0)
    ax4.axhline(y=test_acc, color="purple", linestyle=":", label=f"Test Accuracy ({test_acc:.1f}%)")
    ax4.set_title("Classification Accuracy Dynamics", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy (%)")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="lower right", frameon=True)

    # Panel 5: Normalized Confusion Matrix
    ax5 = plt.subplot(2, 3, 5)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im5 = ax5.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax5.set_title(f"Normalized Confusion Matrix (Acc: {test_acc:.1f}%)", fontsize=11, fontweight='bold')
    ticks = [0, 1]
    ax5.set_xticks(ticks)
    ax5.set_xticklabels(CLASSES)
    ax5.set_yticks(ticks)
    ax5.set_yticklabels(CLASSES)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = cm_norm[i, j]
            col = "white" if pct > 0.5 else "black"
            ax5.text(j, i, f"{val}\n({pct:.0%})", ha="center", va="center", color=col, fontweight='bold')
    ax5.set_ylabel("True Condition")
    ax5.set_xlabel("Predicted Condition")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    # Panel 6: ROC Curve
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(fpr_list, tpr_list, color='#1f77b4', linewidth=2.5, label=f"MaskNet ROC (AUC = {auc_score:.4f})")
    ax6.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label="Random Chance (AUC = 0.50)")
    ax6.scatter([0.0], [1.0], color='green', s=90, zorder=5, label="Ideal Classifier (100% TPR, 0% FPR)")
    ax6.set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight='bold')
    ax6.set_xlabel("False Positive Rate (FPR)")
    ax6.set_ylabel("True Positive Rate (TPR)")
    ax6.grid(True, linestyle=":", alpha=0.6)
    ax6.legend(loc="lower right", fontsize=8, frameon=True)

    report_path = os.path.join(os.path.dirname(__file__), "outputs", "face_mask_detection_report.png")
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Diagnostic report successfully saved to: {report_path}")

    # 7. Serialize Metrics
    metrics_path = os.path.join(os.path.dirname(__file__), "outputs", "mask_detection_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "model_architecture": {
                "name": "MaskNet",
                "parameters": total_params,
                "model_size_kb": model_size_kb,
                "input_resolution": [3, IMG_SIZE, IMG_SIZE],
                "mean_crop_latency_ms": round(mean_lat_ms, 2),
                "fps_throughput": round(1000.0 / max(mean_lat_ms, 0.01), 1)
            },
            "training": {
                "epochs": epochs,
                "duration_seconds": round(train_duration, 2),
                "best_val_accuracy": round(best_val_acc, 2)
            },
            "test_performance": {
                "accuracy_pct": round(test_acc, 2),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(auc_score, 4),
                "confusion_matrix": cm.tolist()
            }
        }, f, indent=4)
    print(f"      Metrics serialized to: {metrics_path}")
    print("\n[SUCCESS] Project 68 Pipeline Complete!\n")

if __name__ == "__main__":
    main()
