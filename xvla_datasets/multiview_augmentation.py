"""Sample-consistent photometric augmentation for synchronized camera views.

Input: a list of RGB ``PIL.Image`` objects for one timestep, ordered exactly as
the dataset camera list (head, left wrist, right wrist for RoboDojo).

Output: a list of normalized ``FloatTensor[3, 224, 224]`` objects in the same
order.  Geometry is never changed.  One category and one set of global color
parameters are sampled per timestep, so all views share environment lighting;
only the 10% sensor branch adds small independent per-camera noise/exposure.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class MultiViewAugmentationParameters:
    mode: str
    scale: float
    brightness: float
    contrast: float
    saturation: float
    hue: float
    gamma: float
    temperature: float
    operation_order: tuple[str, ...]
    sensor_exposure: tuple[float, ...]
    sensor_noise_sigma: tuple[float, ...]

    def to_dict(self):
        value = asdict(self)
        value["operation_order"] = list(self.operation_order)
        value["sensor_exposure"] = list(self.sensor_exposure)
        value["sensor_noise_sigma"] = list(self.sensor_noise_sigma)
        return value


class MultiViewPhotometricAugmentation:
    """Apply the confirmed 50% identity / 40% sync / 10% sensor mixture.

    ``step_value`` may be a multiprocessing.Value shared with DataLoader
    workers.  The optimizer process updates ``step_value.value``; workers read
    it to warm augmentation strength from ``start_scale`` to 1.0.  When it is
    omitted, tests/preview can call :meth:`set_step` locally.
    """

    _OPS = ("brightness", "contrast", "saturation", "hue", "gamma")

    def __init__(
        self,
        *,
        identity_prob: float = 0.5,
        sync_global_prob: float = 0.4,
        sync_sensor_prob: float = 0.1,
        warmup_steps: int = 500,
        start_scale: float = 0.25,
        step_value=None,
        image_size: tuple[int, int] = (224, 224),
    ):
        probabilities = (identity_prob, sync_global_prob, sync_sensor_prob)
        if any(p < 0 for p in probabilities) or abs(sum(probabilities) - 1.0) > 1e-8:
            raise ValueError(f"augmentation probabilities must be non-negative and sum to 1: {probabilities}")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if not 0.0 <= start_scale <= 1.0:
            raise ValueError("start_scale must be in [0,1]")
        self.identity_prob = float(identity_prob)
        self.sync_global_prob = float(sync_global_prob)
        self.sync_sensor_prob = float(sync_sensor_prob)
        self.warmup_steps = int(warmup_steps)
        self.start_scale = float(start_scale)
        self.step_value = step_value
        self.image_size = tuple(image_size)
        self._local_step = 0
        self.last_parameters: MultiViewAugmentationParameters | None = None

    def set_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("augmentation step must be >= 0")
        self._local_step = int(step)

    def current_step(self) -> int:
        return int(self.step_value.value) if self.step_value is not None else self._local_step

    def augmentation_scale(self, step: int | None = None) -> float:
        step = self.current_step() if step is None else int(step)
        if self.warmup_steps == 0:
            return 1.0
        progress = min(max(step, 0) / self.warmup_steps, 1.0)
        return self.start_scale + (1.0 - self.start_scale) * progress

    @staticmethod
    def _toward_one(value: float, scale: float) -> float:
        return 1.0 + scale * (value - 1.0)

    def sample_parameters(self, num_views: int) -> MultiViewAugmentationParameters:
        if num_views <= 0:
            raise ValueError("num_views must be positive")
        draw = random.random()
        if draw < self.identity_prob:
            mode = "identity"
        elif draw < self.identity_prob + self.sync_global_prob:
            mode = "sync_global"
        else:
            mode = "sync_plus_sensor"
        scale = self.augmentation_scale()
        if mode == "identity":
            params = MultiViewAugmentationParameters(
                mode=mode, scale=scale,
                brightness=1.0, contrast=1.0, saturation=1.0,
                hue=0.0, gamma=1.0, temperature=0.0,
                operation_order=self._OPS,
                sensor_exposure=(1.0,) * num_views,
                sensor_noise_sigma=(0.0,) * num_views,
            )
            self.last_parameters = params
            return params

        order = list(self._OPS)
        random.shuffle(order)
        brightness = self._toward_one(random.uniform(0.6, 1.4), scale)
        contrast = self._toward_one(random.uniform(0.7, 1.3), scale)
        saturation = self._toward_one(random.uniform(0.6, 1.4), scale)
        hue = scale * random.uniform(-0.05, 0.05)
        gamma = self._toward_one(random.uniform(0.75, 1.35), scale)
        temperature = scale * random.uniform(-0.15, 0.15)
        if mode == "sync_plus_sensor":
            exposure = tuple(
                self._toward_one(random.uniform(0.95, 1.05), scale)
                for _ in range(num_views)
            )
            sigma = tuple(scale * random.uniform(0.003, 0.015) for _ in range(num_views))
        else:
            exposure = (1.0,) * num_views
            sigma = (0.0,) * num_views
        params = MultiViewAugmentationParameters(
            mode=mode, scale=scale,
            brightness=brightness, contrast=contrast, saturation=saturation,
            hue=hue, gamma=gamma, temperature=temperature,
            operation_order=tuple(order),
            sensor_exposure=exposure, sensor_noise_sigma=sigma,
        )
        self.last_parameters = params
        return params

    @staticmethod
    def _apply_pil_operations(image: Image.Image, params: MultiViewAugmentationParameters):
        operations = {
            "brightness": lambda x: TF.adjust_brightness(x, params.brightness),
            "contrast": lambda x: TF.adjust_contrast(x, params.contrast),
            "saturation": lambda x: TF.adjust_saturation(x, params.saturation),
            "hue": lambda x: TF.adjust_hue(x, params.hue),
            "gamma": lambda x: TF.adjust_gamma(x, params.gamma),
        }
        for name in params.operation_order:
            image = operations[name](image)
        return image

    def apply_with_parameters(
        self,
        images: Sequence[Image.Image],
        params: MultiViewAugmentationParameters,
    ) -> list[torch.Tensor]:
        if not images:
            raise ValueError("at least one camera image is required")
        if len(images) != len(params.sensor_exposure):
            raise ValueError(
                f"parameter view count={len(params.sensor_exposure)} != images={len(images)}"
            )
        outputs = []
        temperature_gain = torch.tensor(
            [1.0 + params.temperature, 1.0, 1.0 - params.temperature],
            dtype=torch.float32,
        ).view(3, 1, 1)
        for index, image in enumerate(images):
            if not isinstance(image, Image.Image):
                raise TypeError(f"expected PIL.Image, got {type(image).__name__}")
            image = TF.resize(image.convert("RGB"), self.image_size, InterpolationMode.BICUBIC)
            if params.mode != "identity":
                image = self._apply_pil_operations(image, params)
            tensor = TF.to_tensor(image)
            tensor = tensor * temperature_gain
            tensor = tensor * params.sensor_exposure[index]
            sigma = params.sensor_noise_sigma[index]
            if sigma > 0:
                tensor = tensor + torch.randn_like(tensor) * sigma
            tensor = tensor.clamp_(0.0, 1.0)
            outputs.append(TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD, inplace=True))
        return outputs

    def __call__(self, images: Sequence[Image.Image]) -> list[torch.Tensor]:
        params = self.sample_parameters(len(images))
        return self.apply_with_parameters(images, params)
