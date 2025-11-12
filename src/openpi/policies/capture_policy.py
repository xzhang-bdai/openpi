import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Convert an arbitrary array or tensor-like image into uint8 HWC layout."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3 and image.ndim == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class CaptureInputs(transforms.DataTransformFn):
    """Map the single capture camera view into the model's three camera slots.

    The capture dataset contains a single RGB stream which we treat as the left wrist view.
    We therefore zero-pad the base and right wrist views while also clearing their masks so
    downstream components can tell those slots are synthetic.
    """

    model_type: _model.ModelType
    camera_key: str = "image"

    def __call__(self, data: dict) -> dict:
        observation_key = f"observation/{self.camera_key}"
        if observation_key not in data:
            raise KeyError(f"Expected {observation_key} in data keys: {list(data.keys())}")

        left_image = _parse_image(data[observation_key])
        base_image = np.zeros_like(left_image)
        right_image = np.zeros_like(left_image)

        image_dict = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": left_image,
            "right_wrist_0_rgb": right_image,
        }
        image_mask = {
            "base_0_rgb": np.False_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,
        }

        inputs = {
            "state": data["observation/state"],
            "image": image_dict,
            "image_mask": image_mask,
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CaptureOutputs(transforms.DataTransformFn):
    """Pass through the first 7 action dimensions (delta cartesian + gripper)."""

    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}

