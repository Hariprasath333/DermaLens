import torch
import numpy as np
import wandb
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backend.classifier.config import DEVICE, EPOCHS, BATCH_SIZE, OUTPUT_DIR
from backend.classifier.models import LesionIQHybrid
from backend.classifier.dataloader import get_dataloaders
from backend.classifier.train import train

# Known training class distribution
counts = {
    0: 3640,  # MEL
    1: 10304, # NV
    2: 2668,  # BCC
    3: 1500,  # AK
    4: 2126,  # BKL
    5: 588,   # DF
    6: 615,   # VASC
    7: 1305   # SCC
}

def compute_class_weights(counts_dict):
    total = sum(counts_dict.values())
    num_classes = len(counts_dict)
    weights = np.zeros(num_classes, dtype=np.float32)
    for cls, count in counts_dict.items():
        weights[cls] = total / (num_classes * count)
    return weights

def main():
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    class_weights = compute_class_weights(counts)
    
    ABLATION_MODES = ['image_only', 'full']  # effnet_only & swin_only already done
    
    # ensure output dir exists
    output_dir = Path(OUTPUT_DIR)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    for mode in ABLATION_MODES:
        print(f"\n{'='*50}")
        print(f"Starting Experiment: {mode.upper()}")
        print(f"{'='*50}")

        model = LesionIQHybrid(num_classes=8, meta_dim=13, mode=mode).to(DEVICE)

        wandb.init(
            project='LesionIQ',
            name=f'ablation_{mode}',
            config={'mode': mode, 'epochs': EPOCHS}
        )

        try:
            best_path = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                epochs=EPOCHS,
                device=DEVICE
            )
            print(f"Successfully finished {mode}. Best checkpoint: {best_path}")
        except Exception as e:
            print(f"Failed during experiment {mode}: {e}")
        finally:
            wandb.finish()
            # Rename best checkpoint so it doesn't get overwritten by next mode
            best_ckpt = output_dir / "checkpoints" / "best_model.pt"
            if best_ckpt.exists():
                renamed = best_ckpt.with_name(f"best_{mode}.pt")
                if renamed.exists():
                    renamed.unlink()
                best_ckpt.rename(renamed)
            # Also rename SWA checkpoint
            swa_ckpt = output_dir / "checkpoints" / "best_model_swa.pt"
            if swa_ckpt.exists():
                swa_renamed = swa_ckpt.with_name(f"best_{mode}_swa.pt")
                if swa_renamed.exists():
                    swa_renamed.unlink()
                swa_ckpt.rename(swa_renamed)

if __name__ == "__main__":
    main()
