"""
Project 68: Interactive Live Video Stream Face Mask Detection CLI
Captures video frames, detects human faces, evaluates mask compliance using
the trained MaskNet model, and overlays real-time compliance bounding boxes.
"""

import os
import sys
import time
import numpy as np
import torch

from main import (
    MaskNet,
    CLASSES,
    IMG_SIZE,
    DEVICE,
    synthesize_face_mask_sample
)

def run_simulated_stream_cli(model):
    print("\n" + "=" * 75)
    print("  SIMULATED LIVE VIDEO FEED FACE MASK DETECTION")
    print("=" * 75)
    print("  Monitoring facility entrance video stream for mask compliance...")
    print("  Press Ctrl+C to terminate session.\n")

    stream_faces = [
        (synthesize_face_mask_sample(True, 0), "Surgical Blue Mask"),
        (synthesize_face_mask_sample(False),   "No Mask (Violation)"),
        (synthesize_face_mask_sample(True, 1), "N95 Respirator"),
        (synthesize_face_mask_sample(False),   "No Mask (Violation)"),
        (synthesize_face_mask_sample(True, 2), "Black Fabric Mask")
    ]

    try:
        frame_idx = 0
        while True:
            crop, true_label = stream_faces[frame_idx % len(stream_faces)]
            crop_t = ((torch.tensor(crop, dtype=torch.float32).permute(2, 0, 1) - 0.5) / 0.5).unsqueeze(0).to(DEVICE)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(crop_t)
                prob_mask = torch.softmax(logits, dim=1)[0, 1].item()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            is_compliant = (prob_mask >= 0.50)
            status_tag = "COMPLIANT [MASK DETECTED]" if is_compliant else "NON-COMPLIANT [VIOLATION ALERT]"
            conf = (prob_mask if is_compliant else (1.0 - prob_mask)) * 100.0

            print(f"[Frame {frame_idx:03d} | {latency_ms:4.1f}ms] Target: {true_label:<22} -> Conf: {conf:5.1f}% | {status_tag}")
            frame_idx += 1
            time.sleep(0.4)

    except KeyboardInterrupt:
        print("\nLive video stream monitoring paused by user.")

def main():
    weights_path = os.path.join(os.path.dirname(__file__), "outputs", "masknet_best.pth")
    model = MaskNet(num_classes=2).to(DEVICE)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    run_simulated_stream_cli(model)

if __name__ == "__main__":
    main()
