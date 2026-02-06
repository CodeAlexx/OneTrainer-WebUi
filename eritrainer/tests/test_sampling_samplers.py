"""Tests for native EriTrainer sampler implementations."""

from __future__ import annotations

from types import SimpleNamespace

from eritrainer.core.interfaces import ModelType
from eritrainer.sampling.sampler import (
    ChromaSampler,
    FluxFillSampler,
    HiDreamSampler,
    HunyuanVideoSampler,
    PixArtSampler,
    QwenImageEditSampler,
    SanaSampler,
    SD3Sampler,
    SDDepthSampler,
    SDInpaintingSampler,
    SDXLInpaintingSampler,
    SDXLSampler,
    WuerstchenSampler,
    create_sampler,
)

from PIL import Image


class _DummyPipeline:
    last_from_pretrained: tuple[str, dict] | None = None
    last_call_kwargs: dict | None = None

    @classmethod
    def from_pretrained(cls, model_source: str, **kwargs):
        cls.last_from_pretrained = (model_source, kwargs)
        return cls()

    def to(self, _device):
        return self

    def __call__(
        self,
        prompt=None,
        negative_prompt=None,
        image=None,
        height=None,
        width=None,
        num_inference_steps=50,
        guidance_scale=None,
        true_cfg_scale=None,
        generator=None,
        output_type="pil",
        return_dict=True,
        **kwargs,
    ):
        captured = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "true_cfg_scale": true_cfg_scale,
            "generator": generator,
            "output_type": output_type,
            "return_dict": return_dict,
        }
        captured.update(kwargs)
        type(self).last_call_kwargs = captured
        return SimpleNamespace(images=[Image.new("RGB", (64, 64), "white")])


def test_sampler_factory_aliases():
    assert isinstance(create_sampler("sd3.5"), SD3Sampler)
    assert isinstance(create_sampler("sd35"), SD3Sampler)
    assert isinstance(create_sampler("sdxl"), SDXLSampler)
    assert isinstance(create_sampler("sd15_inpainting"), SDInpaintingSampler)
    assert isinstance(create_sampler("sd20_depth"), SDDepthSampler)
    assert isinstance(create_sampler("sdxl_inpainting"), SDXLInpaintingSampler)
    assert isinstance(create_sampler("flux_fill_dev"), FluxFillSampler)
    assert isinstance(create_sampler("pixart_sigma"), PixArtSampler)
    assert isinstance(create_sampler("wuerstchen_2"), WuerstchenSampler)
    assert isinstance(create_sampler("sana"), SanaSampler)
    assert isinstance(create_sampler("hunyuan_video"), HunyuanVideoSampler)
    assert isinstance(create_sampler("hi_dream_full"), HiDreamSampler)
    assert isinstance(create_sampler("chroma"), ChromaSampler)
    assert isinstance(create_sampler("qwen_image_edit"), QwenImageEditSampler)


def test_sdxl_sampler_uses_local_files_only_and_quantizes_resolution(monkeypatch, tmp_path):
    model_dir = tmp_path / "sdxl_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    def _load_class(_name: str):
        return _DummyPipeline

    monkeypatch.setattr("eritrainer.sampling.sampler._load_pipeline_class", _load_class)

    sampler = SDXLSampler(model={"path": str(model_dir)}, model_type=ModelType.SDXL)
    image = sampler.sample(
        prompt="a cat",
        height=513,
        width=513,
        num_inference_steps=2,
        guidance_scale=5.0,
        seed=42,
        device="cpu",
        dtype="float32",
    )

    assert isinstance(image, Image.Image)
    assert _DummyPipeline.last_from_pretrained is not None
    _, kwargs = _DummyPipeline.last_from_pretrained
    assert kwargs["local_files_only"] is True

    assert _DummyPipeline.last_call_kwargs is not None
    assert _DummyPipeline.last_call_kwargs["height"] == 512
    assert _DummyPipeline.last_call_kwargs["width"] == 512


def test_qwen_edit_sampler_passes_image(monkeypatch, tmp_path):
    model_dir = tmp_path / "qwen_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    image_path = tmp_path / "input.png"
    Image.new("RGB", (32, 32), "black").save(image_path)

    def _load_class(_name: str):
        return _DummyPipeline

    monkeypatch.setattr("eritrainer.sampling.sampler._load_pipeline_class", _load_class)

    sampler = QwenImageEditSampler(model={"path": str(model_dir)}, model_type=ModelType.QWEN_IMAGE_EDIT)
    sampler.sample(
        prompt="edit this image",
        image_path=image_path,
        num_inference_steps=1,
        guidance_scale=1.5,
        device="cpu",
        dtype="float32",
    )

    assert _DummyPipeline.last_call_kwargs is not None
    assert "image" in _DummyPipeline.last_call_kwargs
    assert isinstance(_DummyPipeline.last_call_kwargs["image"], Image.Image)
    assert _DummyPipeline.last_call_kwargs.get("true_cfg_scale") == 1.5


def test_flux2_klein_prefers_klein_pipeline(monkeypatch, tmp_path):
    model_dir = tmp_path / "flux2_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    loaded_names: list[str] = []

    def _load_class(name: str):
        loaded_names.append(name)
        return _DummyPipeline

    monkeypatch.setattr("eritrainer.sampling.sampler._load_pipeline_class", _load_class)

    sampler = create_sampler(ModelType.FLUX_2_KLEIN_4B, model={"path": str(model_dir)})
    sampler.sample(
        prompt="portrait",
        height=512,
        width=512,
        num_inference_steps=1,
        guidance_scale=4.0,
        device="cpu",
        dtype="float32",
    )

    assert loaded_names
    assert loaded_names[0] == "Flux2KleinPipeline"


def test_pixart_sigma_prefers_sigma_pipeline(monkeypatch, tmp_path):
    model_dir = tmp_path / "pixart_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    loaded_names: list[str] = []

    def _load_class(name: str):
        loaded_names.append(name)
        return _DummyPipeline

    monkeypatch.setattr("eritrainer.sampling.sampler._load_pipeline_class", _load_class)

    sampler = create_sampler(ModelType.PIXART_SIGMA, model={"path": str(model_dir)})
    sampler.sample(
        prompt="portrait",
        height=512,
        width=512,
        num_inference_steps=1,
        guidance_scale=4.0,
        device="cpu",
        dtype="float32",
    )

    assert loaded_names
    assert loaded_names[0] == "PixArtSigmaPipeline"
