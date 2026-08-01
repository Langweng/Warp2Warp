from __future__ import annotations

import argparse
import logging
import os
import shlex
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from datasets.warp2_fix_online_dataset import Warp2FixOnlineDataset
from datasets.warp_fix_dataset import WarpFixDataset
from evaluation.myevaluate import Evaluator
from utils.fixed_test_set import generate_fixed_homography_labels, resolve_fixed_labels_path
from models.base_model_factory import build_base_model
from utils.io import load_config, log_experiment_config, save_config, setup_logger
from utils.seed import set_seed


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Homography checkpoint evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (YAML)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the checkpoint to evaluate")
    parser.add_argument("--device", type=str, default=None, help="Override config device, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to evaluate, e.g. val or test")
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch index used by self-supervised evaluator stage logic; defaults to checkpoint epoch if available",
    )
    parser.add_argument(
        "--fixed-labels",
        type=str,
        default=None,
        help="Path to a pre-generated fixed evaluation labels JSON file.",
    )
    parser.add_argument(
        "--generate-fixed-labels",
        type=str,
        default=None,
        help="Generate fixed evaluation labels at this path before evaluation, then use them for the run.",
    )
    parser.add_argument(
        "--generate-fixed-labels-only",
        action="store_true",
        help="Generate the fixed evaluation labels and exit without running model evaluation.",
    )
    return parser.parse_args()


def resolve_split(config: dict, cli_split: str | None) -> str:
    if cli_split is not None:
        return cli_split
    return config.get("validation", {}).get("split", "val")


def resolve_manifest_path(config: dict, split: str) -> str:
    data_cfg = config["data"]
    split_manifest_key = f"{split}_manifest"
    manifest_path = data_cfg.get(split_manifest_key)
    if manifest_path is None and split == "test":
        manifest_path = data_cfg.get("val_manifest")
    if manifest_path is None:
        raise ValueError(
            f"Unable to resolve manifest for split {split!r}. "
            f"Expected data.{split_manifest_key} in the config."
        )
    return manifest_path


def build_eval_dataset(config: dict, split: str, fixed_labels_path: str | None = None):
    strategy = config["training"].get("strategy", "self-supervised")
    manifest_path = resolve_manifest_path(config, split)
    image_root = config["data"]["image_root"]

    if strategy == "supervised":
        return WarpFixDataset(
            manifest_path=manifest_path,
            image_root=image_root,
            config=config,
            dataset_type=split,
            fixed_labels_path=fixed_labels_path,
        )

    if strategy == "self-supervised":
        return Warp2FixOnlineDataset(
            manifest_path=manifest_path,
            image_root=image_root,
            config=config,
            dataset_type=split,
            fixed_labels_path=fixed_labels_path,
        )

    raise ValueError(
        f"Unsupported training strategy: {strategy}. "
        "Supported strategies are 'supervised' and 'self-supervised'."
    )


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> tuple[dict, dict]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    logger.info("Loaded model weights from %s", os.path.abspath(checkpoint_path))
    return checkpoint, state_dict


def main():
    args = parse_args()
    if args.generate_fixed_labels_only and args.generate_fixed_labels is None:
        raise ValueError("--generate-fixed-labels-only requires --generate-fixed-labels PATH.")

    config = load_config(args.config)

    if args.device is not None:
        config["training"]["device"] = args.device
        config.setdefault("validation", {})["device"] = args.device

    split = resolve_split(config, args.split)
    manifest_path = resolve_manifest_path(config, split)
    fixed_labels_path = resolve_fixed_labels_path(config, args.fixed_labels)
    if args.generate_fixed_labels is not None:
        fixed_labels_path = args.generate_fixed_labels

    exp_name = config["experiment"]["exp_name"]
    log_dir = os.path.join("logs", exp_name, "eval")
    os.makedirs(log_dir, exist_ok=True)
    log_path = setup_logger(log_dir, exp_name="test")
    run_name = os.path.splitext(os.path.basename(log_path))[0]
    save_config(config, os.path.join(log_dir, f"{run_name}_config.yaml"))
    runtime_args = {key: value for key, value in vars(args).items() if value is not None}
    log_experiment_config(
        config,
        config_source=args.config,
        runtime_args=runtime_args,
        command=" ".join(shlex.quote(arg) for arg in sys.argv),
    )

    set_seed(config["training"].get("seed"))

    if args.generate_fixed_labels is not None:
        generated_path = generate_fixed_homography_labels(
            manifest_path=manifest_path,
            split=split,
            config=config,
            output_path=args.generate_fixed_labels,
        )
        logger.info("Generated fixed evaluation labels at %s", os.path.abspath(generated_path))
        fixed_labels_path = generated_path
        if args.generate_fixed_labels_only:
            return

    checkpoint_path = (
        args.checkpoint
        or config.get("testing", {}).get("checkpoint")
        or config.get("training", {}).get("resume")
    )
    if checkpoint_path is None:
        raise ValueError("No checkpoint specified. Use --checkpoint or set testing.checkpoint in config.")

    device = torch.device(config["training"]["device"])
    logger.info("Using device: %s", device)

    model = build_base_model(config).to(device)
    checkpoint, state_dict = load_model_checkpoint(model, checkpoint_path, device)
    model.eval()

    eval_epoch = args.epoch
    if eval_epoch is None:
        eval_epoch = int(checkpoint.get("epoch", sum(config["training"].get("epoch_list", [1]))))
    logger.info("Evaluator epoch context: %d", eval_epoch)

    dataset = build_eval_dataset(config, split, fixed_labels_path=fixed_labels_path)
    data_loader = DataLoader(
        dataset,
        batch_size=config["validation"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=True,
        drop_last=False,
    )
    logger.info("Evaluating split %s with %d samples", split, len(dataset))

    net_copy = None
    if config["training"].get("strategy", "self-supervised") == "self-supervised":
        net_copy = build_base_model(config).to(device)
        net_copy.load_state_dict(state_dict, strict=True)
        net_copy.eval()
        logger.info("Loaded mirrored net_copy from the same checkpoint for self-supervised evaluation")

    evaluator = Evaluator(
        model,
        device,
        config,
        net_copy=net_copy,
        current_epoch=eval_epoch,
        report_dir=log_dir,
        report_prefix=run_name,
    )
    evaluator.set_epoch(eval_epoch)

    with torch.no_grad():
        metrics = evaluator.evaluate(data_loader)

    strategy = config["training"].get("strategy", "self-supervised")
    if strategy == "supervised":
        mean_mace, _ = metrics
        logger.info("Final Mean MACE: %.6f", mean_mace)
    else:
        mean_mace_refine, mean_mace_warp2_fix, mean_mace_warp_fix, *_ = metrics
        logger.info("Final Refine Mean MACE: %.6f", mean_mace_refine)
        logger.info("Final Warp2-Fix Mean MACE: %.6f", mean_mace_warp2_fix)
        logger.info("Final Warp-Fix Mean MACE: %.6f", mean_mace_warp_fix)

    if fixed_labels_path is not None:
        logger.info("Fixed evaluation labels: %s", os.path.abspath(fixed_labels_path))

    detailed_report = evaluator.last_detailed_report
    if detailed_report is not None:
        metric_key = detailed_report["metric_key"]
        for group_name in ("easy", "moderate", "hard"):
            group_info = detailed_report["group_metrics"][group_name]
            if group_info["mace"] is None:
                logger.info(
                    "%s group (%s) %s: no samples",
                    group_name.capitalize(),
                    group_info["range"],
                    metric_key,
                )
            else:
                logger.info(
                    "%s group (%s) %s: %.6f over %d samples",
                    group_name.capitalize(),
                    group_info["range"],
                    metric_key,
                    group_info["mace"],
                    group_info["count"],
                )
        if detailed_report["csv_path"] is not None:
            logger.info("Per-sample metrics CSV saved to %s", os.path.abspath(detailed_report["csv_path"]))


if __name__ == "__main__":
    main()
