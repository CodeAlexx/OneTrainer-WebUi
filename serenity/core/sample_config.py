"""Sample generation configuration.

Defines ``SampleConfig`` for controlling periodic sample generation
during training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from serenity.core.enums import NoiseScheduler


@dataclass
class SampleConfig:
    """Configuration for a single sample generation specification.

    Instances are typically stored in a list within TrainConfig so that
    multiple prompts / configurations can be sampled during training.
    """

    # Whether this sample spec is active
    enabled: bool = True

    # Prompt / generation settings
    prompt: str = ""
    negative_prompt: str = ""
    height: int = 512
    width: int = 512
    seed: int = 42
    random_seed: bool = False
    num_inference_steps: int = 20
    guidance_scale: float = 7.0
    noise_scheduler: NoiseScheduler | str = NoiseScheduler.EULER

    # Video-specific
    frames: int = 1
    length: float = 10.0

    # Scheduling
    sample_every_n_steps: int = 0
    sample_every_n_epochs: int = 0

    # Output
    output_format: str = "png"  # png, jpg, webp
    output_dir: str = ""

    # Text encoder overrides
    text_encoder_1_layer_skip: int = 0
    text_encoder_2_layer_skip: int = 0

    # Inpainting
    sample_inpainting: bool = False
    base_image_path: str = ""
    mask_image_path: str = ""

    def __post_init__(self) -> None:
        """Normalize enum-ish sample fields."""
        if not isinstance(self.noise_scheduler, NoiseScheduler):
            raw = str(self.noise_scheduler)
            try:
                self.noise_scheduler = NoiseScheduler(raw)
            except ValueError:
                upper = raw.upper()
                for member in NoiseScheduler:
                    if member.value.upper() == upper:
                        self.noise_scheduler = member
                        break
                else:
                    self.noise_scheduler = NoiseScheduler.EULER

    def should_sample_at_step(self, global_step: int) -> bool:
        """Check if a sample should be generated at this step."""
        if not self.enabled:
            return False
        if self.sample_every_n_steps <= 0:
            return False
        return global_step > 0 and (global_step % self.sample_every_n_steps == 0)

    def should_sample_at_epoch(self, epoch: int) -> bool:
        """Check if a sample should be generated at this epoch."""
        if not self.enabled:
            return False
        if self.sample_every_n_epochs <= 0:
            return False
        return epoch > 0 and (epoch % self.sample_every_n_epochs == 0)

    def get_seed(self) -> int:
        """Return the seed to use, optionally randomized."""
        if self.random_seed:
            import random
            return random.randint(0, 2**32 - 1)
        return self.seed


@dataclass
class SampleSchedule:
    """Collection of sample configs with schedule management."""

    configs: list[SampleConfig] = field(default_factory=list)

    def configs_for_step(self, global_step: int) -> list[SampleConfig]:
        """Return configs that should generate a sample at this step."""
        return [c for c in self.configs if c.should_sample_at_step(global_step)]

    def configs_for_epoch(self, epoch: int) -> list[SampleConfig]:
        """Return configs that should generate a sample at this epoch."""
        return [c for c in self.configs if c.should_sample_at_epoch(epoch)]

    def any_active(self) -> bool:
        """True if any sample config is enabled."""
        return any(c.enabled for c in self.configs)


__all__ = [
    "SampleConfig",
    "SampleSchedule",
]
