from __future__ import annotations

from typing import Any


def build_base_model(config: dict[str, Any]):
    model_name = config.get("model", {}).get("name")
    if model_name == "IHN":
        from models.ihn_net import IHN

        image_h, image_w = config["experiment"]["image_size"]
        marginal = int(config["data"]["marginal"])
        crop_h = int(image_h) - 2 * marginal
        crop_w = int(image_w) - 2 * marginal
        if crop_h != crop_w:
            raise ValueError(
                f"IHN currently requires square crops, got crop size ({crop_h}, {crop_w})."
            )

        model_cfg = dict(config["model"]["IHN"])
        model_cfg.setdefault("crop_size", crop_h)
        return IHN(config=model_cfg)

    raise ValueError(f"Unsupported model.name={model_name!r}. This clean release includes only 'IHN'.")
