from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from utils.homography_utils import get_homography_with_adaptation, resolve_homography_adaptation


def load_manifest_entries(manifest_path: str, split: str) -> list[dict[str, Any]]:
    with open(manifest_path, "r") as file:
        manifest_data = json.load(file)

    if isinstance(manifest_data, list):
        return manifest_data
    if isinstance(manifest_data, dict) and split in manifest_data:
        return manifest_data.get(split, [])

    raise ValueError(f"Invalid manifest format for split {split!r}: {manifest_path}")


def build_sample_key(item: dict[str, Any], index: int) -> str:
    if "sample_id" in item:
        return str(item["sample_id"])
    if "id" in item:
        return str(item["id"])
    return f"{index}:{item.get('fixA_path', '')}|{item.get('fixB_path', '')}"


def resolve_fixed_labels_path(config: dict, explicit_path: str | None = None) -> str | None:
    if explicit_path:
        return explicit_path
    return config.get("testing", {}).get("fixed_labels_path")


def _validate_delta(name: str, delta: Any) -> np.ndarray:
    array = np.asarray(delta, dtype=np.float32)
    if array.shape != (4, 2):
        raise ValueError(f"{name} must have shape (4, 2), got {array.shape}.")
    return array


def generate_fixed_homography_labels(
    manifest_path: str,
    split: str,
    config: dict,
    output_path: str,
) -> str:
    entries = load_manifest_entries(manifest_path, split)
    adaptation = resolve_homography_adaptation(config)

    image_h = int(config["experiment"]["image_size"][0])
    image_w = int(config["experiment"]["image_size"][1])
    marginal = int(config["data"]["marginal"])
    perturb_range = int(config["data"]["perturb_range"])
    warp2_range = int(config.get("training", {}).get("warp2", 0))

    payload = {
        "metadata": {
            "split": split,
            "manifest_path": manifest_path,
            "image_size": [image_h, image_w],
            "marginal": marginal,
            "perturb_range": perturb_range,
            "warp2": warp2_range,
            "homography_adaptation": adaptation,
        },
        "samples": [],
    }

    for index, item in enumerate(entries):
        _, _, d_true = get_homography_with_adaptation(
            image_h,
            image_w,
            marginal,
            perturb_range,
            adaptation=adaptation,
        )
        _, _, d_eval = get_homography_with_adaptation(
            image_h,
            image_w,
            marginal,
            warp2_range,
            adaptation=adaptation,
        )
        payload["samples"].append(
            {
                "index": index,
                "sample_key": build_sample_key(item, index),
                "fixA_path": item.get("fixA_path"),
                "fixB_path": item.get("fixB_path"),
                "d_true": np.asarray(d_true, dtype=np.float32).tolist(),
                "d_eval": np.asarray(d_eval, dtype=np.float32).tolist(),
            }
        )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(payload, file, indent=2)

    return output_path


def load_fixed_homography_labels(
    fixed_labels_path: str,
    manifest_entries: list[dict[str, Any]],
    split: str,
    config: dict,
) -> list[dict[str, Any]]:
    with open(fixed_labels_path, "r") as file:
        payload = json.load(file)

    records = payload.get("samples")
    if not isinstance(records, list):
        raise ValueError(f"Invalid fixed labels file {fixed_labels_path}: missing top-level 'samples' list.")

    metadata = payload.get("metadata", {})
    expected_image_size = [int(config["experiment"]["image_size"][0]), int(config["experiment"]["image_size"][1])]
    if metadata.get("split") is not None and metadata.get("split") != split:
        raise ValueError(
            f"Fixed labels split mismatch: file has {metadata.get('split')!r}, requested split is {split!r}."
        )
    if metadata.get("image_size") is not None and metadata.get("image_size") != expected_image_size:
        raise ValueError(
            f"Fixed labels image_size mismatch: file has {metadata.get('image_size')!r}, "
            f"expected {expected_image_size!r}."
        )
    if metadata.get("marginal") is not None and metadata.get("marginal") != int(config['data']['marginal']):
        raise ValueError(
            f"Fixed labels marginal mismatch: file has {metadata.get('marginal')!r}, "
            f"expected {int(config['data']['marginal'])!r}."
        )

    if len(records) != len(manifest_entries):
        raise ValueError(
            f"Fixed labels sample count mismatch: file has {len(records)} samples, "
            f"manifest has {len(manifest_entries)} samples."
        )

    normalized_records: list[dict[str, Any]] = []
    for index, (item, record) in enumerate(zip(manifest_entries, records)):
        expected_key = build_sample_key(item, index)
        record_index = int(record.get("index", index))
        if record_index != index:
            raise ValueError(
                f"Fixed labels index mismatch at position {index}: file has {record_index}, expected {index}."
            )
        if record.get("fixA_path") != item.get("fixA_path") or record.get("fixB_path") != item.get("fixB_path"):
            raise ValueError(
                f"Fixed labels manifest mismatch at index {index}: "
                f"expected ({item.get('fixA_path')}, {item.get('fixB_path')}), "
                f"got ({record.get('fixA_path')}, {record.get('fixB_path')})."
            )

        normalized_records.append(
            {
                "index": index,
                "sample_key": str(record.get("sample_key", expected_key)),
                "fixA_path": record.get("fixA_path"),
                "fixB_path": record.get("fixB_path"),
                "d_true": _validate_delta(f"records[{index}].d_true", record.get("d_true")),
                "d_eval": _validate_delta(f"records[{index}].d_eval", record.get("d_eval", np.zeros((4, 2)))),
            }
        )

    return normalized_records
