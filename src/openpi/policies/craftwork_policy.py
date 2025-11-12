import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class CraftworkInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    camera_keys: tuple[str, ...]
    base_camera_index: int = 0
    left_wrist_index: int | None = 1
    right_wrist_index: int | None = 2

    def __call__(self, data: dict) -> dict:
        def _get_camera_image(index: int | None) -> tuple[np.ndarray | None, np.bool_]:
            if index is None:
                return None, np.False_
            if index < 0 or index >= len(self.camera_keys):
                return None, np.False_
            key = self.camera_keys[index]
            obs_key = f"observation/{key}"
            if obs_key not in data:
                return None, np.False_
            return _parse_image(data[obs_key]), np.True_

        base_image, base_mask = _get_camera_image(self.base_camera_index)
        if base_image is None:
            raise KeyError(
                f"Missing base camera index {self.base_camera_index} in data: {list(data.keys())}"
            )

        left_image, left_mask = _get_camera_image(self.left_wrist_index)
        right_image, right_mask = _get_camera_image(self.right_wrist_index)

        if left_image is None:
            left_image = np.zeros_like(base_image)
        if right_image is None:
            right_image = np.zeros_like(base_image)

        image_dict = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": left_image,
            "right_wrist_0_rgb": right_image,
        }
        image_mask = {
            "base_0_rgb": base_mask,
            "left_wrist_0_rgb": left_mask,
            "right_wrist_0_rgb": right_mask,
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
class CraftworkOutputs(transforms.DataTransformFn):
    action_dim: int = 7 # 6D delta cartesian actions + 1D gripper action

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}

