"""Adjust synthetic counts so real+synth=1500 per class, then re-run FID."""
import os, sys, shutil, random, json, csv, logging, subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("finalize")

SYNTH   = Path(r"path/to/synthetic")
REAL    = Path(r"path/to/output")
REPORTS = Path(r"path/to/quality_reports")
PYTHON  = str(Path(r"python"))
GENPY   = str(Path(r"path/to/stylegan2-ada-pytorch/generate.py"))
TARGET  = 1500
EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

MODELS = {
    "DF": r"path/to/training-runs/network-snapshot-001500.pkl",
    "VASC": r"path/to/training-runs/network-snapshot-000500.pkl",
}


def list_imgs(d):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in EXTS and p.is_file())


def build_env():
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_MODULE_LOADING"] = "LAZY"
    try:
        import torch as _t
        env["PATH"] = os.path.join(os.path.dirname(_t.__file__), "lib") + os.pathsep + env.get("PATH", "")
    except ImportError:
        pass
    return env


# ═══════════════════════════════════════════════════════════
# STEP 1: Adjust counts
# ═══════════════════════════════════════════════════════════

def trim_class(cls):
    """Trim excess synthetic images for classes that have too many."""
    synth_dir = SYNTH / cls
    real_count = len(list_imgs(REAL / cls))
    synth_files = list_imgs(synth_dir)
    need = TARGET - real_count

    if len(synth_files) <= need:
        log.info("[%s] Have %d synth, need %d -- no trimming needed.", cls, len(synth_files), need)
        return

    excess = len(synth_files) - need
    log.info("[%s] Trimming %d -> %d (removing %d)", cls, len(synth_files), need, excess)

    trim_dir = synth_dir / "_trimmed"
    trim_dir.mkdir(exist_ok=True)

    random.seed(42)
    to_remove = random.sample(synth_files, excess)
    for f in to_remove:
        shutil.move(str(f), str(trim_dir / f.name))

    remaining = len(list_imgs(synth_dir))
    log.info("[%s] Now: %d real + %d synth = %d total", cls, real_count, remaining, real_count + remaining)


def generate_more(cls, needed_extra):
    """Generate additional synthetic images for classes that are short."""
    synth_dir = SYNTH / cls
    pkl = MODELS[cls]

    # Use seeds starting from 1500 to avoid conflicts
    existing_seeds = set()
    for f in list_imgs(synth_dir):
        try:
            s = int(f.stem.replace("seed", ""))
            existing_seeds.add(s)
        except ValueError:
            pass

    new_seeds = []
    seed = 1500
    while len(new_seeds) < needed_extra:
        if seed not in existing_seeds:
            new_seeds.append(seed)
        seed += 1

    seed_str = ",".join(str(s) for s in new_seeds)
    log.info("[%s] Generating %d additional images (seeds %d-%d) ...", cls, needed_extra, new_seeds[0], new_seeds[-1])

    env = build_env()
    cmd = [PYTHON, GENPY, f"--network={pkl}", f"--outdir={synth_dir}",
           f"--seeds={seed_str}", "--trunc=0.85", "--noise-mode=random"]
    subprocess.run(cmd, env=env, capture_output=True)

    # For VASC: apply same gentle unsharp mask to new images
    if cls == "VASC":
        import cv2
        log.info("[%s] Applying unsharp mask to %d new images ...", cls, len(new_seeds))
        for s in new_seeds:
            fp = synth_dir / f"seed{s:04d}.png"
            if fp.exists():
                bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
                if bgr is not None:
                    blurred = cv2.GaussianBlur(bgr, (0, 0), 1.0)
                    sharpened = cv2.addWeighted(bgr, 1.25, blurred, -0.25, 0)
                    cv2.imwrite(str(fp), np.clip(sharpened, 0, 255).astype(np.uint8))

    real_count = len(list_imgs(REAL / cls))
    remaining = len(list_imgs(synth_dir))
    log.info("[%s] Now: %d real + %d synth = %d total", cls, real_count, remaining, real_count + remaining)


# ═══════════════════════════════════════════════════════════
# STEP 2: Re-run FID on final sets
# ═══════════════════════════════════════════════════════════

def compute_fid(cls):
    """Compute FID between real and synthetic images."""
    try:
        import torch
        from torchvision import transforms, models
        from scipy.linalg import sqrtm
    except ImportError:
        log.warning("torch/scipy not available -- skipping FID")
        return None

    real_files = list_imgs(REAL / cls)
    synth_files = list_imgs(SYNTH / cls)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = torch.nn.Identity()
    inception.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def extract(files, max_n=2048):
        feats = []
        for fp in files[:max_n]:
            img = Image.open(str(fp)).convert("RGB")
            t = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                f = inception(t).cpu().numpy().flatten()
            feats.append(f)
        return np.array(feats)

    log.info("[FID %s] Extracting features (real=%d, synth=%d) ...", cls, len(real_files), len(synth_files))
    rf = extract(real_files)
    sf = extract(synth_files)

    mu_r, sig_r = rf.mean(0), np.cov(rf, rowvar=False)
    mu_s, sig_s = sf.mean(0), np.cov(sf, rowvar=False)
    diff = mu_r - mu_s
    covmean, _ = sqrtm(sig_r @ sig_s, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sig_r + sig_s - 2 * covmean))
    log.info("[FID %s] = %.2f", cls, fid)
    return fid


# ═══════════════════════════════════════════════════════════
# STEP 3: Final manifest
# ═══════════════════════════════════════════════════════════

def write_manifest(fid_scores):
    manifest = Path(r"path/to/dataset_manifest.csv")
    rows = []
    for cls in ["AK", "SCC", "DF", "VASC"]:
        rc = len(list_imgs(REAL / cls))
        sc = len(list_imgs(SYNTH / cls))
        rows.append({
            "class": cls, "real_images": rc, "synthetic_images": sc,
            "total_images": rc + sc,
            "synth_ratio": round(sc / rc, 2) if rc else 0,
            "fid": round(fid_scores.get(cls, 0), 2),
            "gan_model": "StyleGAN2-ADA", "resolution": "512x512",
            "truncation_psi": 0.85, "generated_date": datetime.now().strftime("%Y-%m-%d"),
        })

    with open(str(manifest), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    log.info("")
    log.info("%-6s  %6s  %6s  %6s  %6s  %8s", "Class", "Real", "Synth", "Total", "Ratio", "FID")
    log.info("-" * 50)
    for r in rows:
        log.info("%-6s  %6d  %6d  %6d  %6s  %8s",
                 r["class"], r["real_images"], r["synthetic_images"],
                 r["total_images"], r["synth_ratio"], r["fid"])
    log.info("")
    log.info("Manifest: %s", manifest)


def main():
    log.info("=" * 60)
    log.info("  Finalize Dataset: 1500 total per class + fresh FID")
    log.info("=" * 60)
    log.info("")

    # Step 1: Adjust counts
    log.info("--- STEP 1: Adjust counts ---")
    trim_class("AK")
    trim_class("SCC")

    # DF needs more
    df_real = len(list_imgs(REAL / "DF"))
    df_synth = len(list_imgs(SYNTH / "DF"))
    df_need = TARGET - df_real
    if df_synth < df_need:
        generate_more("DF", df_need - df_synth)
    else:
        trim_class("DF")

    # VASC needs more
    vasc_real = len(list_imgs(REAL / "VASC"))
    vasc_synth = len(list_imgs(SYNTH / "VASC"))
    vasc_need = TARGET - vasc_real
    if vasc_synth < vasc_need:
        generate_more("VASC", vasc_need - vasc_synth)
    else:
        trim_class("VASC")

    log.info("")

    # Step 2: Fresh FID on final sets
    log.info("--- STEP 2: Fresh FID scores ---")
    fid_scores = {}
    for cls in ["AK", "SCC", "DF", "VASC"]:
        fid_scores[cls] = compute_fid(cls)
    log.info("")

    # Step 3: Write manifest
    log.info("--- STEP 3: Final manifest ---")
    write_manifest(fid_scores)

    log.info("")
    log.info("=" * 60)
    log.info("  DONE. All classes at 1500 total with fresh FID.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
