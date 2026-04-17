from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any
import inspect

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import SCOPDepthTextToImageDataset, collate_training_items

DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"


def resolve_torch_device(device_preference: str = "auto") -> str:
    if device_preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device_preference == "cuda" and not torch.cuda.is_available():
        print("Training: CUDA requested but not available, falling back to CPU.")
        return "cpu"

    if (
        device_preference == "mps"
        and not (torch.backends.mps.is_built() and torch.backends.mps.is_available())
    ):
        print("Training: MPS requested but not available, falling back to CPU.")
        return "cpu"

    return device_preference


def choose_weight_dtype(device: str, mixed_precision: str) -> torch.dtype:
    if mixed_precision == "no":
        return torch.float32
    if device == "cuda" and mixed_precision == "fp16":
        return torch.float16
    if device == "cuda" and mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def build_autocast_context(device: str, mixed_precision: str):
    if device == "cuda" and mixed_precision in {"fp16", "bf16"}:
        dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.autocast(device_type="cpu", enabled=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an SD1.5 LoRA baseline on SCOP-Depth."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=("no", "fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--prompt-prefix", type=str, default="a photo of")
    parser.add_argument("--validation-prompts-file", type=Path, default=None)
    parser.add_argument("--num-validation-images", type=int, default=4)
    parser.add_argument("--validation-every", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--disable-tqdm", action="store_true")
    return parser


def _is_tqdm_disabled(args: argparse.Namespace) -> bool:
    return args.disable_tqdm or os.environ.get("TQDM_DISABLE") == "1" or not sys.stderr.isatty()


def _load_validation_prompts(
    dataset: SCOPDepthTextToImageDataset, prompt_file: Path | None
) -> list[str]:
    from .prompts import prompt_from_scop_depth_row

    if prompt_file is not None:
        prompts = [line.strip() for line in prompt_file.read_text().splitlines() if line.strip()]
        if prompts:
            return prompts
    prompts: list[str] = []
    for row in dataset.rows[:4]:
        prompts.append(prompt_from_scop_depth_row(row, prefix=dataset.prompt_prefix))
    return prompts


def _build_lora_attn_procs(unet: Any, rank: int) -> dict[str, Any]:
    from diffusers.models.attention_processor import (
        LoRAAttnProcessor,
        LoRAAttnProcessor2_0,
    )

    attn_processor_cls = (
        LoRAAttnProcessor2_0
        if hasattr(F, "scaled_dot_product_attention")
        else LoRAAttnProcessor
    )

    lora_attn_procs: dict[str, Any] = {}
    block_out_channels = list(unet.config.block_out_channels)
    reversed_block_out_channels = list(reversed(block_out_channels))

    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        )

        if name.startswith("mid_block"):
            hidden_size = block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = reversed_block_out_channels[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = block_out_channels[block_id]
        else:
            raise ValueError(f"Unrecognized attention processor name: {name}")

        lora_attn_procs[name] = attn_processor_cls(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )

    return lora_attn_procs


def _manual_lora_supported() -> bool:
    from diffusers.models.attention_processor import LoRAAttnProcessor2_0

    signature = inspect.signature(LoRAAttnProcessor2_0.__init__)
    return "hidden_size" in signature.parameters


def _attach_lora_adapters(unet: Any, rank: int, learning_rate: float) -> torch.optim.Optimizer:
    if hasattr(unet, "add_adapter"):
        try:
            from peft import LoraConfig
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The training baseline now uses the PEFT LoRA path for compatibility "
                "with this diffusers version. Install the training extras again so "
                "`peft` is available: ./.venv/bin/python -m pip install -e '.[train]'"
            ) from exc

        target_modules = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
        ]
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        unet.add_adapter(lora_config)
        trainable_parameters = [p for p in unet.parameters() if p.requires_grad]
        for parameter in trainable_parameters:
            if parameter.dtype != torch.float32:
                parameter.data = parameter.data.to(torch.float32)
        return torch.optim.AdamW(trainable_parameters, lr=learning_rate)

    if _manual_lora_supported():
        from diffusers.loaders import AttnProcsLayers

        unet.set_attn_processor(_build_lora_attn_procs(unet, rank))
        lora_layers = AttnProcsLayers(unet.attn_processors)
        lora_layers.to(device=next(unet.parameters()).device, dtype=torch.float32)
        return torch.optim.AdamW(lora_layers.parameters(), lr=learning_rate)

    raise RuntimeError(
        "This diffusers installation does not expose a compatible LoRA attachment API."
    )


def _save_training_state(
    output_dir: Path,
    *,
    step: int,
    args: argparse.Namespace,
    prompts: list[str],
) -> None:
    payload = {
        "step": step,
        "dataset_dir": str(args.dataset_dir),
        "model_id": args.model_id,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "validation_prompts": prompts,
    }
    (output_dir / "training_state.json").write_text(json.dumps(payload, indent=2))


def _save_checkpoint(
    *,
    unet: Any,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
) -> Path:
    checkpoint_dir = output_dir / f"checkpoint-{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = checkpoint_dir / "lora"
    if hasattr(unet, "save_lora_adapter"):
        unet.save_lora_adapter(lora_dir, adapter_name="default")
    else:
        unet.save_attn_procs(lora_dir)
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    return checkpoint_dir


def _load_checkpoint_if_requested(
    *,
    args: argparse.Namespace,
    unet: Any,
    optimizer: torch.optim.Optimizer,
) -> int:
    if args.resume_from_checkpoint is None:
        return 0

    checkpoint_dir = args.resume_from_checkpoint
    unet.load_attn_procs(checkpoint_dir / "lora")
    optimizer_path = checkpoint_dir / "optimizer.pt"
    if optimizer_path.exists():
        optimizer.load_state_dict(
            torch.load(optimizer_path, map_location="cpu", weights_only=False)
        )
    name = checkpoint_dir.name
    if name.startswith("checkpoint-"):
        return int(name.split("-", 1)[1])
    return 0


def _run_validation(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    prompts: list[str],
    device: str,
    dtype: torch.dtype,
    unet: Any,
    vae: Any,
    text_encoder: Any,
    tokenizer: Any,
    noise_scheduler: Any,
    step: int,
) -> None:
    from diffusers import StableDiffusionPipeline

    validation_dir = output_dir / "validation" / f"step-{step:06d}"
    validation_dir.mkdir(parents=True, exist_ok=True)

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        torch_dtype=dtype,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)

    for prompt_index, prompt in enumerate(prompts):
        for image_index in range(args.num_validation_images):
            generator = torch.Generator(device="cpu").manual_seed(
                args.seed + step + prompt_index * args.num_validation_images + image_index
            )
            image = pipeline(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]
            image.save(validation_dir / f"prompt{prompt_index:02d}_{image_index:02d}.png")

    del pipeline
    if device == "cuda":
        torch.cuda.empty_cache()


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_torch_device(args.device)
    weight_dtype = choose_weight_dtype(device, args.mixed_precision)
    set_seed(args.seed)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device == "cuda" and args.mixed_precision == "fp16",
    )

    dataset = SCOPDepthTextToImageDataset(
        args.dataset_dir,
        image_size=args.image_size,
        prompt_prefix=args.prompt_prefix,
        limit_rows=args.limit_rows,
        shuffle_rows=True,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_training_items,
    )

    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=weight_dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.model_id,
        subfolder="unet",
        torch_dtype=weight_dtype,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    optimizer = _attach_lora_adapters(unet, args.lora_rank, args.learning_rate)
    unet.to(device)

    start_step = _load_checkpoint_if_requested(
        args=args,
        unet=unet,
        optimizer=optimizer,
    )

    validation_prompts = _load_validation_prompts(dataset, args.validation_prompts_file)
    _save_training_state(args.output_dir, step=start_step, args=args, prompts=validation_prompts)

    progress_bar = tqdm(
        total=args.max_train_steps,
        initial=start_step,
        disable=_is_tqdm_disabled(args),
        desc="Training",
    )

    global_step = start_step
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    while global_step < args.max_train_steps:
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
                dtype=torch.int64,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            text_inputs = tokenizer(
                batch["prompts"],
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                encoder_hidden_states = text_encoder(
                    text_inputs.input_ids.to(device),
                    attention_mask=text_inputs.attention_mask.to(device),
                )[0]

            with build_autocast_context(device, args.mixed_precision):
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Encountered a non-finite training loss. "
                    "Try rerunning with MIXED_PRECISION=no or a smaller learning rate."
                )
            loss = loss / args.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                progress_bar.update(1)

                if global_step % args.log_every == 0:
                    progress_bar.set_postfix(loss=f"{loss.item() * args.gradient_accumulation_steps:.4f}")

                if global_step % args.save_every == 0:
                    checkpoint_dir = _save_checkpoint(
                        unet=unet,
                        optimizer=optimizer,
                        output_dir=args.output_dir,
                        step=global_step,
                    )
                    _save_training_state(
                        args.output_dir,
                        step=global_step,
                        args=args,
                        prompts=validation_prompts,
                    )
                    print(f"Saved checkpoint to {checkpoint_dir}")

                if args.validation_every > 0 and global_step % args.validation_every == 0:
                    _run_validation(
                        args=args,
                        output_dir=args.output_dir,
                        prompts=validation_prompts,
                        device=device,
                        dtype=weight_dtype,
                        unet=unet,
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        noise_scheduler=noise_scheduler,
                        step=global_step,
                    )

                if global_step >= args.max_train_steps:
                    break

        else:
            continue
        break

    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(unet, "save_lora_adapter"):
        unet.save_lora_adapter(final_dir / "lora", adapter_name="default")
    else:
        unet.save_attn_procs(final_dir / "lora")
    _save_training_state(args.output_dir, step=global_step, args=args, prompts=validation_prompts)
    print(f"Training finished at step {global_step}. Final LoRA weights saved to {final_dir / 'lora'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
