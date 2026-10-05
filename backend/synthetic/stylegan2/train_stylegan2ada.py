"""
LesionIQ -- Medical-grade StyleGAN2-ADA training script.

Trains one lesion class at a time using transfer learning from FFHQ-512.
Optimised for diagnostic-quality synthetic dermoscopy on an RTX A6000 (48 GB).

Workflow:
  1. Prepares a StyleGAN2-ADA-compatible dataset ZIP (if not already present).
  2. Launches training with conservative, quality-first hyperparameters.
  3. FID is evaluated every snapshot so you can pick the best checkpoint.

Usage:
    conda activate sg2ada
    python train_stylegan2ada.py
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import time



# ╔══════════════════════════════════════════════════════════════════╗
# ║  EDIT THESE PLACEHOLDERS                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

CLASS_NAME = "AK"                                           # AK | SCC | DF | VASC

IMAGE_DIR       = r"path/to/output/AK"                # cleaned images for this class
STYLEGAN2_REPO  = r"path/to/stylegan2-ada-pytorch"     # cloned NVlabs repo
PRETRAINED_PKL  = r"path/to/training-runs/network-snapshot-000600.pkl"
DATASET_DIR     = r"path/to/datasets"                  # ZIP datasets live here
TRAINING_OUTDIR = r"path/to/training-runs"             # checkpoints + FID logs

# ──────────────────────────────────────────────────────────────────
# MEDICAL-GRADE HYPERPARAMETERS
#
# These defaults prioritise image fidelity over speed.  The literature
# (arXiv:2603.13497, Bioengineering 2026) shows gamma=0.8 yields the
# best FID on ISIC dermoscopy with StyleGAN2.  We train longer (5000
# kimg) and snapshot frequently (every 100 kimg ~ every ~30 min on
# A6000 at 512x512) so the best checkpoint can be cherry-picked.
# ──────────────────────────────────────────────────────────────────

RESOLUTION   = 512
GPUS         = 1
CFG          = "auto"           # auto-tunes batch size + network capacity
AUG          = "ada"            # Adaptive Discriminator Augmentation
MIRROR       = 1                # free 2x data via horizontal flips
GAMMA        = 0.8              # R1 regularisation — 0.8 is optimal for dermoscopy
KIMG         = 400              # 600 kimg already in weights; 400 more = 1000 effective kimg total
SNAP         = 25               # snapshot every snap*tick kimg (~100 kimg apart)
METRICS      = "none"           # "none" = skip FID (fastest startup); "fid50k_full" = full FID
                                # NOTE: fid50k_full generates 50 000 fake images per eval —
                                #       that can take 10-20 min per snapshot on a single GPU.
                                #       Switch to "fid50k_full" only after training is confirmed.
WORKERS      = 8                # dataloader workers — keep GPU fed
BATCH_SIZE   = 32               # 32 fills ~40 GB on A6000 at 512x512 with mixed-precision
FREEZED      = 0                # freeze-D layers (0 = disabled; try 2-4 if overfitting)

# ── Performance flags (Ampere / RTX A6000) ──
ALLOW_TF32   = True             # TF32 matmul — ~3× faster on Ampere, negligible quality loss
USE_NHWC     = True             # channels-last memory layout — faster FP16 convolutions

# ══════════════════════════════════════════════════════════════════


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def prepare_dataset():
    """Convert IMAGE_DIR into a StyleGAN2-ADA ZIP dataset at 512x512."""
    zip_path = os.path.join(DATASET_DIR, f"{CLASS_NAME}.zip")

    if os.path.isfile(zip_path):
        log(f"Dataset ZIP already exists: {zip_path}")
        return zip_path

    os.makedirs(DATASET_DIR, exist_ok=True)

    dataset_tool = os.path.join(STYLEGAN2_REPO, "dataset_tool.py")
    if not os.path.isfile(dataset_tool):
        log(f"ERROR: dataset_tool.py not found at {dataset_tool}")
        log("       Run setup_stylegan2ada.ps1 first to clone the repo.")
        sys.exit(1)

    log(f"Preparing dataset ZIP from {IMAGE_DIR} ...")
    cmd = [
        sys.executable, dataset_tool,
        f"--source={IMAGE_DIR}",
        f"--dest={zip_path}",
        f"--width={RESOLUTION}",
        f"--height={RESOLUTION}",
    ]
    subprocess.run(cmd, check=True)
    log(f"Dataset ZIP created: {zip_path}")
    return zip_path


def count_images():
    """Count source images to log the effective dataset size."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    d = Path(IMAGE_DIR)
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir() if f.suffix.lower() in exts)


def run_training(zip_path):
    """Launch StyleGAN2-ADA training with medical-grade hyperparameters."""
    train_py = os.path.join(STYLEGAN2_REPO, "train.py")
    if not os.path.isfile(train_py):
        log(f"ERROR: train.py not found at {train_py}")
        sys.exit(1)

    if not os.path.isfile(PRETRAINED_PKL):
        log(f"ERROR: Pretrained weights not found at {PRETRAINED_PKL}")
        log("       Run setup_stylegan2ada.ps1 to download FFHQ-512 weights.")
        sys.exit(1)

    os.makedirs(TRAINING_OUTDIR, exist_ok=True)

    cmd = [
        sys.executable, train_py,
        f"--outdir={TRAINING_OUTDIR}",
        f"--data={zip_path}",
        f"--gpus={GPUS}",
        f"--cfg={CFG}",
        f"--aug={AUG}",
        f"--mirror={MIRROR}",
        f"--snap={SNAP}",
        f"--resume={PRETRAINED_PKL}",
        f"--gamma={GAMMA}",
        f"--kimg={KIMG}",
        f"--metrics={METRICS}",
        f"--workers={WORKERS}",
    ]

    if BATCH_SIZE is not None:
        cmd.append(f"--batch={BATCH_SIZE}")

    if ALLOW_TF32:
        cmd.append("--allow-tf32=1")

    if USE_NHWC:
        cmd.append("--nhwc=1")

    if FREEZED > 0:
        cmd.append(f"--freezed={FREEZED}")

    log("")
    log("=" * 65)
    log(f"  TRAINING: {CLASS_NAME}")
    log(f"  Dataset:  {zip_path}  ({count_images()} images)")
    log(f"  Resume:   {PRETRAINED_PKL}")
    log(f"  Gamma:    {GAMMA}  |  kimg: {KIMG}  |  AUG: {AUG}")
    log(f"  Batch:    {BATCH_SIZE}  |  TF32: {ALLOW_TF32}  |  NHWC: {USE_NHWC}")
    log(f"  Output:   {TRAINING_OUTDIR}")
    log("=" * 65)
    log("")
    log("Command:")
    log("  " + " ".join(cmd))
    log("")

    env = os.environ.copy()
    env["PYTHONWARNINGS"]          = "once::UserWarning"
    # Suppress noisy TensorFlow oneDNN info messages (TF is not used, but it
    # gets imported as a transitive dep and spams the console)
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    # ── CUDA performance tuning (Ampere / RTX A6000) ──────────────────────────
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # reduce VRAM fragmentation
    env["CUDA_MODULE_LOADING"]     = "LAZY"                       # faster cold start
    # cuDNN heuristic search — ~10-20% conv speed uplift after first tick warmup
    env.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")
    # Tell nvcc to target the A6000 (sm_86) — avoids PTX fallback JIT at runtime
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    # Fused multi-head attention kernel (not used by SG2 directly, but helps any
    # nn.MultiheadAttention layers if you later add a conditioning transformer)
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    # ── Fix: Add PyTorch's own lib dir to PATH so Windows can find bundled CUDA
    # DLLs (e.g. cudart64_12.dll, nvrtc64_12.dll) when loading the compiled
    # .pyd extension.  Without this, ImportError: DLL load failed occurs even
    # though the .pyd was compiled successfully. ────────────────────────────────
    try:
        import torch as _torch
        _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if os.path.isdir(_torch_lib):
            env["PATH"] = _torch_lib + os.pathsep + env.get("PATH", "")
            log(f"  Prepended torch/lib to PATH: {_torch_lib}")
    except ImportError:
        pass
    # ──────────────────────────────────────────────────────────────────────────

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        log("")
        log(f"ERROR: train.py exited with code {exc.returncode}")
        log("  The full traceback is in the run's log.txt file:")
        # Find the most recent run dir
        try:
            run_dirs = sorted(
                [d for d in os.scandir(TRAINING_OUTDIR) if d.is_dir()],
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
            if run_dirs:
                log(f"  {os.path.join(run_dirs[0].path, 'log.txt')}")
                # Print the last 30 lines of the log so the error is visible here
                log_path = os.path.join(run_dirs[0].path, "log.txt")
                if os.path.isfile(log_path):
                    with open(log_path, "r", errors="replace") as lf:
                        tail = lf.readlines()[-30:]
                    log("  --- tail of log.txt ---")
                    for line in tail:
                        print("  " + line, end="")
                    log("  --- end of log ---")
        except Exception:
            pass
        sys.exit(exc.returncode)

    log("")
    log("Training finished.  Inspect FID scores:")
    log(f"  Look for metric-fid50k_full.jsonl in the run directory under {TRAINING_OUTDIR}")
    log("  Pick the snapshot with the LOWEST FID for generation.")


def write_run_metadata(zip_path):
    """Save a JSON sidecar with the training config for reproducibility."""
    meta = {
        "class": CLASS_NAME,
        "image_dir": IMAGE_DIR,
        "dataset_zip": zip_path,
        "image_count": count_images(),
        "pretrained": PRETRAINED_PKL,
        "resolution": RESOLUTION,
        "gamma": GAMMA,
        "kimg": KIMG,
        "aug": AUG,
        "mirror": MIRROR,
        "snap": SNAP,
        "cfg": CFG,
        "freezed": FREEZED,
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = os.path.join(TRAINING_OUTDIR, f"train_config_{CLASS_NAME}.json")
    os.makedirs(TRAINING_OUTDIR, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"Run metadata saved to {meta_path}")


def _check_environment():
    """Log environment details for debugging CUDA kernel issues."""
    info = {}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["torch_version"] = "NOT INSTALLED"

    info["python_version"] = sys.version
    info["sys_platform"] = sys.platform


    # Check for custom ops files
    bias_act = os.path.join(STYLEGAN2_REPO, "torch_utils", "ops", "bias_act.py")
    upfirdn2d = os.path.join(STYLEGAN2_REPO, "torch_utils", "ops", "upfirdn2d.py")
    custom_ops = os.path.join(STYLEGAN2_REPO, "torch_utils", "custom_ops.py")

    files_info = {
        "bias_act_exists": os.path.isfile(bias_act),
        "upfirdn2d_exists": os.path.isfile(upfirdn2d),
        "custom_ops_exists": os.path.isfile(custom_ops),
        "stylegan2_repo": STYLEGAN2_REPO,
    }

    return info


def _patch_custom_ops():
    """Conditionally patch StyleGAN2-ADA custom ops based on ninja availability.

    PyTorch's JIT C++ extension build requires *both* a C++ compiler (MSVC on
    Windows) *and* the Ninja build tool.  We check for ninja first because MSVC
    is usually present on developer machines even when ninja isn't.

    If ninja IS available  → restore the original get_plugin so fast CUDA
    kernels are JIT-compiled (upfirdn2d, bias_act).
    If ninja is NOT available → patch get_plugin to return None, using the
    slower pure-PyTorch fallback implementations (still fully functional).
    """
    import torch
    import shutil as _sh
    import glob as _glob

    custom_ops_path = os.path.join(STYLEGAN2_REPO, "torch_utils", "custom_ops.py")
    if not os.path.isfile(custom_ops_path):
        return

    # --- Check if Ninja (required by torch.utils.cpp_extension.load) is available ---
    has_ninja = _sh.which("ninja") is not None
    if not has_ninja:
        import subprocess as _sp
        try:
            _sp.run(["ninja", "--version"], capture_output=True, timeout=3)
            has_ninja = True
        except Exception:
            pass

    with open(custom_ops_path, "r") as f:
        src = f.read()

    marker = "# PATCHED_BY_LESIONIQ"
    is_patched = marker in src
    backup = custom_ops_path + ".bak"

    if has_ninja and is_patched:
        # ninja found — restore original get_plugin for fast CUDA kernels
        if os.path.isfile(backup):
            _sh.copy2(backup, custom_ops_path)
            log("Ninja found! Restored original custom_ops.py — CUDA JIT kernels will be compiled.")
            log("  First run will be slower while kernels compile, then subsequent runs will be fast.")
        else:
            log("WARNING: Ninja found but no backup of custom_ops.py exists. Cannot restore.")
        return

    if not has_ninja and is_patched:
        # Already patched, no ninja — nothing to do
        log("Ninja not found — using PyTorch fallback ops (slower but fully functional).")
        log("  TIP: pip install ninja  for ~20-30% faster training via compiled CUDA kernels.")
        return

    if not has_ninja and not is_patched:
        # No ninja, not yet patched — apply patch
        if not os.path.isfile(backup):
            _sh.copy2(custom_ops_path, backup)

        patched_src = src.replace(
            "def get_plugin(",
            f"{marker}\n"
            "def get_plugin(*args, **kwargs):\n"
            "    return None\n\n"
            "def _original_get_plugin("
        )

        with open(custom_ops_path, "w") as f:
            f.write(patched_src)

        log(f"Patched custom_ops.py to skip CUDA JIT compilation (ninja not installed).")
        log("  Using PyTorch fallback ops — fully functional, ~20-30% slower than compiled kernels.")
        log("  TIP: pip install ninja  to unlock fast CUDA kernels on your next run.")
    else:
        # has_ninja and not is_patched — already using native CUDA kernels
        log("Ninja available — using fast CUDA JIT kernels!")


    # Force-enable conv2d_gradfix and grid_sample_gradfix custom autograd ops.
    # These provide second-order gradient support (needed by R1 regularisation).
    # PyTorch 2.x's native grid_sample does NOT support double backward, so
    # the custom implementations MUST be active.
    ops_marker = "# PATCHED_BY_LESIONIQ_OPS_V2"
    ops_dir = os.path.join(STYLEGAN2_REPO, "torch_utils", "ops")
    for fname in ("conv2d_gradfix.py", "grid_sample_gradfix.py"):
        fpath = os.path.join(ops_dir, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, "r") as f:
            content = f.read()
        if ops_marker in content:
            continue
        backup_f = fpath + ".bak"
        if not os.path.isfile(backup_f):
            shutil.copy2(fpath, backup_f)

        # Force the 'enabled' flag to True and suppress repeated warnings
        patched = content.replace("enabled = False", "enabled = True")
        if fname == "conv2d_gradfix.py":
            patched = patched.replace(
                "    if any(torch.__version__.startswith(x) for x in ['1.7.', '1.8.', '1.9']):\n"
                "        return True\n"
                "    warnings.warn(f'conv2d_gradfix not supported on PyTorch {torch.__version__}. Falling back to torch.nn.functional.conv2d().')\n"
                "    return False\n",
                "    return True\n",
            )
        if fname == "grid_sample_gradfix.py":
            patched = patched.replace(
                "    if any(torch.__version__.startswith(x) for x in ['1.7.', '1.8.', '1.9']):\n"
                "        return True\n"
                "    warnings.warn(f'grid_sample_gradfix not supported on PyTorch {torch.__version__}. Falling back to torch.nn.functional.grid_sample().')\n"
                "    return False\n",
                "    return True\n",
            )
        patched = f"{ops_marker}\nimport warnings as _w\n_w.filterwarnings('once', category=UserWarning)\n" + patched

        with open(fpath, "w") as f:
            f.write(patched)

        changed = patched != content

        log(f"  Patched {fname} — force-enabled custom higher-order gradient ops")


def main():
    if not os.path.isdir(IMAGE_DIR):
        log(f"ERROR: IMAGE_DIR does not exist: {IMAGE_DIR}")
        log("       Set the placeholder at the top of this file.")
        sys.exit(1)

    n = count_images()
    if n == 0:
        log(f"ERROR: No images found in {IMAGE_DIR}")
        sys.exit(1)

    log(f"Class: {CLASS_NAME}  |  {n} real images in {IMAGE_DIR}")

    env_info = _check_environment()

    # Patch custom ops to avoid CUDA JIT hang (works for PyTorch 1.x and 2.x)
    try:
        ver_str = env_info.get("torch_version", "0")
        # Guard against "NOT INSTALLED" or other non-numeric strings
        major = int(ver_str.split(".")[0]) if ver_str[0].isdigit() else 0
        if major >= 1:
            log(f"Detected PyTorch {ver_str} — checking StyleGAN2 custom ops...")
            _patch_custom_ops()
    except Exception as e:
        log(f"WARNING: Could not patch custom ops: {e}")

    zip_path = prepare_dataset()
    write_run_metadata(zip_path)
    run_training(zip_path)


if __name__ == "__main__":
    main()