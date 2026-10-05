"""
LesionIQ Research Pipeline — single entry point.

Usage:
    python run.py verify    --data-root <dir>
    python run.py preprocess --data-root <dir> --out-root <dir>
    python run.py split     --pre-root <dir> --raw-root <dir> --out <dir>
                            --datasets isic2019 ham10000 ...
    python run.py train     --variant V1 --split-dir <dir> --out-dir <dir>
    python run.py evaluate  --variant V1 --checkpoint <best.pt>
                            --split-dir <dir> --out-dir <dir>
    python run.py audit     --variant V1 --checkpoint <best.pt>
                            --split-dir <dir> --out-dir <dir>
    python run.py ensemble  --checkpoints v1.pt v2.pt ...
                            --variants V1 V2 ... --split-dir <dir>
                            --out-dir <dir>
    python run.py selftest                     [no args; runs synthetic E2E]
    python run.py full      --config pipeline.yaml

The `full` subcommand reads pipeline.yaml and runs verify->preprocess->
split->train->evaluate->audit end-to-end.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_verify(args):
    from stages.stage1_datasets import verify_datasets, print_status_report
    present, missing = verify_datasets(args.data_root,
                                        only_tier=args.only_tier)
    print_status_report(present, missing, Path(args.data_root),
                        show_download_instructions=not args.quiet)
    return 1 if any(s.spec.tier == 1 for s in missing) else 0


def cmd_preprocess(args):
    from stages.stage2_preprocess import preprocess_dataset
    from stages.stage1_datasets import DATASET_REGISTRY, verify_dataset
    data_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    selected = args.datasets or list(DATASET_REGISTRY.keys())
    for key in selected:
        spec = DATASET_REGISTRY.get(key)
        if spec is None:
            logging.error("Unknown dataset key: %s", key); continue
        status = verify_dataset(spec, data_root)
        if not status.present:
            logging.warning("Skipping %s — not verified.", key); continue
        preprocess_dataset(spec, data_root, out_root,
                            resize=args.resize, workers=args.workers,
                            skip_if_exists=not args.no_skip_existing)
    return 0


def cmd_split(args):
    from stages.stage3_split import run_split
    out_dir = run_split(
        pre_root=Path(args.pre_root).expanduser().resolve(),
        raw_root=Path(args.raw_root).expanduser().resolve(),
        out_root=Path(args.out).expanduser().resolve(),
        datasets=args.datasets, mode=args.mode, seed=args.seed,
        dedup=not args.no_dedup,
    )
    print(f"Split output: {out_dir}")
    return 0


def cmd_train(args):
    from stages.stage6_train import train, TrainConfig
    cfg = TrainConfig(
        variant_id=args.variant, split_dir=args.split_dir,
        out_dir=args.out_dir, epochs=args.epochs,
        batch_size=args.batch_size, img_size=args.img_size,
        num_workers=args.num_workers, lr=args.lr, loss=args.loss,
        use_timm=not args.no_timm, pretrained=not args.no_pretrained,
        seed=args.seed,
    )
    best = train(cfg)
    print(f"Best checkpoint: {best}")
    return 0


def cmd_evaluate(args):
    from stages.stage7_evaluate import evaluate
    metrics = evaluate(
        variant_id=args.variant, checkpoint_path=args.checkpoint,
        split_dir=args.split_dir, out_dir=args.out_dir,
        img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers, use_timm=not args.no_timm,
    )
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_audit(args):
    from stages.stage8_audit import run_audit
    res = run_audit(
        variant_id=args.variant, checkpoint_path=args.checkpoint,
        split_dir=args.split_dir, out_dir=args.out_dir,
        img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers, use_timm=not args.no_timm,
    )
    print(json.dumps({"variant": args.variant,
                      "n_test": res.get("n_test"),
                      "audit_keys": list(res.keys())}, indent=2))
    return 0


def cmd_ensemble(args):
    """Logit-space weighted ensemble across a list of trained variants."""
    from stages.stage5_ensemble import (
        EnsembleWrapper, fit_ensemble_weights, predict_calibrated,
        evaluate_ensemble,
    )
    if len(args.checkpoints) != len(args.variants):
        print("ERROR: --checkpoints and --variants must have same length")
        return 2
    res = evaluate_ensemble(
        variant_ids=args.variants, checkpoint_paths=args.checkpoints,
        split_dir=args.split_dir, out_dir=args.out_dir,
        img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers, use_timm=not args.no_timm,
    )
    print(json.dumps(res, indent=2))
    return 0


def _check_metadata_gradient_flow(use_timm: bool = False,
                                  bootstrap_steps: int = 3) -> None:
    """Regression guard: assert EVERY variant CAN route gradient to `meta`.

    This is the test that would have caught the token-injection silent
    no-op (Bug B): a variant that discards metadata produces no gradient
    on the meta input no matter how long it trains, and fails loudly here
    instead of silently training image-only.

    Note on bootstrapping: FiLM / conditional-BN variants are *identity-
    initialized* (zero-init scale/shift projections) for training
    stability, so at step 0 they legitimately produce zero gradient to
    the meta input — the projection *weights* receive gradient first,
    and only after they move does gradient reach the meta input. We
    therefore take a few optimizer steps before the check. A genuinely
    broken variant (discarded meta) never develops meta gradient, so it
    still fails.
    """
    import torch
    from models import build_variant, ALL_VARIANT_IDS

    log_g = logging.getLogger("lesioniq.selftest")
    img = torch.randn(4, 3, 96, 96)
    failures = []
    for vid in ALL_VARIANT_IDS:
        model = build_variant(vid, use_timm=use_timm).train()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        # Bootstrap past identity-init so projection weights become nonzero.
        for _ in range(bootstrap_steps):
            meta = torch.randn(4, 19)
            mask = torch.ones(4, 19)
            target = torch.randint(0, 8, (4,))
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(img, meta, mask), target)
            loss.backward()
            opt.step()
        # Now check: does the meta INPUT receive gradient?
        meta = torch.randn(4, 19, requires_grad=True)
        mask = torch.ones(4, 19)
        model(img, meta, mask).sum().backward()
        g = 0.0 if meta.grad is None else float(meta.grad.abs().sum())
        if g == 0.0:
            failures.append(vid)
            log_g.error("  %s: meta-grad ZERO — metadata pathway is dead", vid)
        else:
            log_g.info("  %s: meta-grad |sum|=%.4g  OK", vid, g)
    if failures:
        raise AssertionError(
            f"Variants that IGNORE metadata (zero gradient to meta input "
            f"after {bootstrap_steps} steps): {failures}. Their fusion "
            f"mechanism is a silent no-op.")
    log_g.info("Gradient-flow guard PASSED for all %d variants",
               len(ALL_VARIANT_IDS))


def _check_ema_restore() -> None:
    """Regression guard for Bug A: EMA evaluation must not corrupt the
    live training weights.

    Asserts store()->copy_to()->restore() returns the model to its exact
    pre-evaluation weights, and that the shadow remains intact for reuse.
    """
    import torch
    from stages.stage6_train import EMA

    model = torch.nn.Linear(8, 8)
    # Diverge the shadow from the live weights so copy_to is observable.
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))      # live weights move away from shadow
    ema.update(model)                         # shadow now a blend (still != live)

    live_before = {k: v.clone() for k, v in model.state_dict().items()}

    ema.store(model)
    ema.copy_to(model)
    # During eval the model should hold the shadow, not the live weights.
    swapped = any(not torch.equal(model.state_dict()[k], live_before[k])
                  for k in live_before)
    ema.restore(model)

    restored_ok = all(torch.equal(model.state_dict()[k], live_before[k])
                      for k in live_before)
    if not swapped:
        raise AssertionError("EMA.copy_to did not swap in the shadow weights")
    if not restored_ok:
        raise AssertionError(
            "EMA.restore did not return the live weights — Bug A would "
            "corrupt the training trajectory.")
    # Shadow must survive the round-trip for the next epoch.
    ema.store(model); ema.copy_to(model); ema.restore(model)
    logging.getLogger("lesioniq.selftest").info(
        "EMA store/restore invariant PASSED")


def cmd_selftest(args):
    """End-to-end smoke test on synthetic data — no real datasets needed.

    Runs in two parts:
      1. Gradient-flow guard — every variant V0..V11 must route gradient
         to its metadata input (regression test for the token-injection
         silent no-op).
      2. Full pipeline — tiny synthetic split, train V0 1 epoch with the
         tiny-backbone fallback, evaluate, audit.
    """
    import tempfile, os, cv2, numpy as np, pandas as pd

    # Part 1: metadata gradient-flow guard (fast, no I/O)
    logging.getLogger("lesioniq.selftest").info(
        "Running metadata gradient-flow guard for all variants ...")
    _check_metadata_gradient_flow(use_timm=False)

    # Part 1b: EMA store/restore invariant (regression test for Bug A)
    logging.getLogger("lesioniq.selftest").info(
        "Checking EMA store/restore invariant ...")
    _check_ema_restore()
    from stages.stage6_train import train, TrainConfig
    from stages.stage7_evaluate import evaluate
    from stages.stage8_audit import run_audit

    with tempfile.TemporaryDirectory() as td:
        td = os.path.abspath(td)
        img_dir = os.path.join(td, "images")
        os.makedirs(img_dir, exist_ok=True)
        rng = np.random.default_rng(0)
        rows = []
        for i in range(40):
            ipath = os.path.join(img_dir, f"syn_{i}.jpg")
            cv2.imwrite(ipath, (rng.random((96, 96, 3)) * 255).astype(np.uint8))
            cls = i % 8
            rows.append(dict(
                image_id=f"syn_{i}",
                src_image_path=ipath,
                class_name=["MEL","NV","BCC","AK","BKL","DF","VASC","SCC"][cls],
                class_idx=cls,
                lesion_id=f"L{i//2:03d}",
                age=40 + i, sex="female" if i % 3 == 0 else "male",
                site="head/neck", fitzpatrick=(i % 6) + 1,
            ))
        df = pd.DataFrame(rows)
        split_dir = os.path.join(td, "split")
        os.makedirs(split_dir, exist_ok=True)
        for name in ("train", "val_select", "val_calibrate", "test"):
            df.to_csv(os.path.join(split_dir, f"{name}.csv"), index=False)

        out = os.path.join(td, "run")
        # EMA + MixUp ON so the new store/restore + select-and-save path
        # and the no-meta-mix path are both exercised end-to-end.
        cfg = TrainConfig(
            variant_id="V8", split_dir=split_dir, out_dir=out,
            epochs=2, batch_size=8, img_size=64, num_workers=0,
            use_timm=False, pretrained=False,
            use_ema=True, ema_decay=0.9, use_amp=False, mixup_alpha=0.2,
        )
        best = train(cfg)

        # best.pt must record which weights were saved and reload cleanly.
        import torch
        ckpt = torch.load(str(best), map_location="cpu", weights_only=False)
        assert "weights_source" in ckpt, "best.pt missing weights_source tag"
        assert ckpt["weights_source"] in ("live", "ema")
        from models import build_variant
        m = build_variant("V8", use_timm=False)
        m.load_state_dict(ckpt["model"], strict=False)
        logging.getLogger("lesioniq.selftest").info(
            "best.pt reload OK (weights_source=%s)", ckpt["weights_source"])

        evaluate("V8", str(best), split_dir, out, img_size=64,
                  batch_size=8, num_workers=0, use_timm=False)
        run_audit("V8", str(best), split_dir, out, img_size=64,
                   batch_size=8, num_workers=0, use_timm=False)

    print("SELF-TEST PASSED: gradient-guard + EMA-invariant + "
          "train(EMA,MixUp) -> evaluate -> audit all complete.")
    return 0


def cmd_full(args):
    import yaml
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    data_root = cfg["data_root"]
    pre_root = cfg["pre_root"]
    splits_root = cfg["splits_root"]
    runs_root = cfg["runs_root"]
    datasets = cfg.get("datasets", ["isic2019"])
    variants = cfg.get("variants", ["V0"])

    # 1. verify
    from stages.stage1_datasets import verify_datasets
    present, missing = verify_datasets(data_root, only_tier=1)
    if missing and any(s.spec.tier == 1 for s in missing):
        print("Tier-1 datasets missing. Run `python run.py verify --data-root ...` first.")
        return 2

    # 2. preprocess
    from stages.stage2_preprocess import preprocess_dataset
    from stages.stage1_datasets import DATASET_REGISTRY, verify_dataset
    for key in datasets:
        spec = DATASET_REGISTRY[key]
        if verify_dataset(spec, Path(data_root)).present:
            preprocess_dataset(spec, Path(data_root), Path(pre_root),
                                resize=cfg.get("resize", 384),
                                workers=cfg.get("preprocess_workers", 4))

    # 3. split
    from stages.stage3_split import run_split
    split_dir = run_split(
        pre_root=Path(pre_root), raw_root=Path(data_root),
        out_root=Path(splits_root), datasets=datasets,
        mode=cfg.get("split_mode", "single"),
        seed=cfg.get("seed", 42), dedup=True,
    )

    # 4. for each variant: train + evaluate
    from stages.stage6_train import train, TrainConfig
    from stages.stage7_evaluate import evaluate
    summary = {}
    for v in variants:
        run_dir = Path(runs_root) / v
        run_dir.mkdir(parents=True, exist_ok=True)
        tc = TrainConfig(
            variant_id=v, split_dir=str(split_dir), out_dir=str(run_dir),
            epochs=cfg.get("epochs", 30),
            batch_size=cfg.get("batch_size", 32),
            img_size=cfg.get("img_size", 384),
            num_workers=cfg.get("num_workers", 4),
            lr=cfg.get("lr", 1e-4),
            loss=cfg.get("loss", "focal"),
            use_timm=cfg.get("use_timm", True),
            pretrained=cfg.get("pretrained", True),
            seed=cfg.get("seed", 42),
        )
        best = train(tc)
        metrics = evaluate(v, str(best), str(split_dir), str(run_dir),
                            img_size=tc.img_size,
                            batch_size=tc.batch_size,
                            num_workers=tc.num_workers,
                            use_timm=tc.use_timm)
        summary[v] = {"checkpoint": str(best), "metrics": metrics}

    summary_path = Path(runs_root) / "SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"All done. Summary: {summary_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="lesioniq",
                                 description="LesionIQ Research Pipeline")
    p.add_argument("--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--data-root", required=True)
    p_verify.add_argument("--only-tier", type=int, default=None)
    p_verify.add_argument("--quiet", action="store_true")
    p_verify.set_defaults(func=cmd_verify)

    p_pre = sub.add_parser("preprocess")
    p_pre.add_argument("--data-root", required=True)
    p_pre.add_argument("--out-root", required=True)
    p_pre.add_argument("--datasets", nargs="+", default=None)
    p_pre.add_argument("--resize", type=int, default=384)
    p_pre.add_argument("--workers", type=int, default=4)
    p_pre.add_argument("--no-skip-existing", action="store_true")
    p_pre.set_defaults(func=cmd_preprocess)

    p_split = sub.add_parser("split")
    p_split.add_argument("--pre-root", required=True)
    p_split.add_argument("--raw-root", required=True)
    p_split.add_argument("--out", required=True)
    p_split.add_argument("--datasets", nargs="+", required=True)
    p_split.add_argument("--mode", choices=["single", "kfold"], default="single")
    p_split.add_argument("--seed", type=int, default=42)
    p_split.add_argument("--no-dedup", action="store_true")
    p_split.set_defaults(func=cmd_split)

    p_train = sub.add_parser("train")
    p_train.add_argument("--variant", required=True)
    p_train.add_argument("--split-dir", required=True)
    p_train.add_argument("--out-dir", required=True)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--img-size", type=int, default=384)
    p_train.add_argument("--num-workers", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--loss", default="focal",
                          choices=["focal", "cb_focal", "soft_f1", "ldam"])
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--no-timm", action="store_true")
    p_train.add_argument("--no-pretrained", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--variant", required=True)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--split-dir", required=True)
    p_eval.add_argument("--out-dir", required=True)
    p_eval.add_argument("--img-size", type=int, default=384)
    p_eval.add_argument("--batch-size", type=int, default=32)
    p_eval.add_argument("--num-workers", type=int, default=4)
    p_eval.add_argument("--no-timm", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    p_audit = sub.add_parser("audit",
                              help="Fairness + per-lesion + selective + missing-meta + attribution")
    p_audit.add_argument("--variant", required=True)
    p_audit.add_argument("--checkpoint", required=True)
    p_audit.add_argument("--split-dir", required=True)
    p_audit.add_argument("--out-dir", required=True)
    p_audit.add_argument("--img-size", type=int, default=384)
    p_audit.add_argument("--batch-size", type=int, default=32)
    p_audit.add_argument("--num-workers", type=int, default=4)
    p_audit.add_argument("--no-timm", action="store_true")
    p_audit.set_defaults(func=cmd_audit)

    p_ens = sub.add_parser("ensemble",
                            help="Logit-space weighted ensemble of trained variants")
    p_ens.add_argument("--variants", nargs="+", required=True)
    p_ens.add_argument("--checkpoints", nargs="+", required=True)
    p_ens.add_argument("--split-dir", required=True)
    p_ens.add_argument("--out-dir", required=True)
    p_ens.add_argument("--img-size", type=int, default=384)
    p_ens.add_argument("--batch-size", type=int, default=32)
    p_ens.add_argument("--num-workers", type=int, default=4)
    p_ens.add_argument("--no-timm", action="store_true")
    p_ens.set_defaults(func=cmd_ensemble)

    p_selftest = sub.add_parser("selftest",
                                  help="Synthetic E2E smoke test (no real datasets)")
    p_selftest.set_defaults(func=cmd_selftest)

    p_full = sub.add_parser("full",
                            help="End-to-end run from YAML config.")
    p_full.add_argument("--config", required=True)
    p_full.set_defaults(func=cmd_full)

    args = p.parse_args()
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
