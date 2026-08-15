import argparse
import gc
import os
from datetime import datetime

import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, set_seed
from diffusers import AutoencoderKL, FluxTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src import method_adagrpo, method_pavegrpo
from utils import utilities
from utils.dataset import CaptionDataset, PromptEncoder, caption_collate_function, load_captions
from utils.rewards import build_reward_fn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "pretrained_models")

METHODS = {"PaveGRPO": method_pavegrpo, "AdaGRPO": method_adagrpo}

LORA_TARGET_MODULES = [
    "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",
]


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    method = METHODS[args.method]

    deepspeed_plugin = None
    if args.train_mode == "full":
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=2,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clipping=args.max_grad_norm,
        )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    print(f"Rank {local_rank}: Pre-loading reward model '{args.reward_name}' ...")
    reward_fn = build_reward_fn(args, device)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_dir=args.output_dir,
        deepspeed_plugin=deepspeed_plugin,
    )

    def main_print(msg):
        if accelerator.is_main_process:
            accelerator.print(msg)

    log_txt_path = os.path.join(args.output_dir, "train_log.txt")

    def main_write(log_str):
        if accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(log_txt_path, "a", encoding="utf-8") as f:
                f.write(log_str)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = os.path.join(
            args.output_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(log_path, "w") as f:
            f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\nArguments:\n")
            for key, value in vars(args).items():
                f.write(f"  {key}: {value}\n")
            f.write("=" * 50 + "\n\n")

    train_dtype = torch.float32 if args.mixed_precision == "no" else (
        torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16)
    infer_dtype = torch.bfloat16

    # ---------------- Model loading ----------------
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        torch_dtype=train_dtype)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        torch_dtype=infer_dtype).to(device)
    vae.requires_grad_(False)
    vae.enable_tiling()
    image_processor = VaeImageProcessor(vae_scale_factor=16)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.train_mode == "lora":
        main_print(f"--> Training Mode: LoRA (Rank: {args.lora_rank})")
        transformer = get_peft_model(transformer, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            init_lora_weights="gaussian", lora_dropout=0.0, bias="none"))
    elif args.train_mode == "full":
        main_print("--> Training Mode: Full Fine-tuning")
        transformer.train()
    else:
        raise ValueError(f"Unknown train_mode: {args.train_mode}")

    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, transformer.parameters())),
        lr=args.learning_rate, betas=(0.9, 0.999),
        weight_decay=args.weight_decay, eps=1e-8)

    # ---------------- Data: plain-text prompts + on-the-fly encoding ----------------
    captions = load_captions(args.prompt_txt_path)
    main_print(f"[TextEncode] Loaded {len(captions)} prompts from {args.prompt_txt_path} "
               f"(on-the-fly lazy encoding, cache={args.encode_cache_size})")
    prompt_encoder = PromptEncoder(args.pretrained_model_name_or_path, device,
                                   dtype=infer_dtype, cache_size=args.encode_cache_size)
    train_dataloader = DataLoader(
        CaptionDataset(captions),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=caption_collate_function,
        num_workers=args.dataloader_num_workers,
        pin_memory=True, drop_last=True)

    lr_scheduler = utilities.get_warmup_constant_anneal_scheduler(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_anneal_steps=args.num_anneal_steps,
        num_training_steps=args.max_train_steps)

    transformer, optimizer, train_dataloader = accelerator.prepare(
        transformer, optimizer, train_dataloader)
    main_print(f"--> Accelerator Initialized. Distributed Type: {accelerator.distributed_type}")

    progress_bar = tqdm(train_dataloader, total=len(train_dataloader),
                        disable=not accelerator.is_local_main_process, desc="Training")

    sigma_scheduler = torch.linspace(1, 0, args.denoising_steps + 1,
                                     dtype=torch.float32, device=device)
    sigma_scheduler = (args.shift * sigma_scheduler) / (1 + (args.shift - 1) * sigma_scheduler)
    guidance = torch.full([1], args.guidance_scale, device=device, dtype=torch.float32)

    utilities.assert_valid_sequence(args.sde_stepidx, args.denoising_steps)

    global_step = 0
    ema_avg, ema_var = None, None
    ema_history_path = os.path.join(args.output_dir, "ema_history.txt")

    # ---------------- Training loop ----------------
    for _, batch in enumerate(progress_bar):
        if global_step >= args.max_train_steps:
            accelerator.wait_for_everyone()
            break
        with accelerator.accumulate(transformer):
            captions_batch = batch["captions"]
            encoder_hidden_states, pooled_prompt_embeds = prompt_encoder.encode(captions_batch)

            if args.method == "AdaGRPO":
                sel = method.sample_evaluate_and_select_ema(
                    args, transformer, vae, encoder_hidden_states, pooled_prompt_embeds,
                    captions_batch, guidance, sigma_scheduler, image_processor,
                    reward_fn, ema_avg, ema_var)
                ema_avg, ema_var = sel["new_ema_avg"], sel["new_ema_var"]
                ema_std = max(ema_var, 1e-8) ** 0.5
                anchor_results = sel["anchor_results"]
                group_results = sel["group_results"]
                encoder_hidden_states = sel["encoder_hidden_states"]
                pooled_prompt_embeds = sel["pooled_prompt_embeds"]
                captions_batch = sel["captions"]
                main_write(f"ema: step={global_step} mean={ema_avg:.4f} std={ema_std:.4f} "
                           f"selected_idx={sel['selected_idx']}\n")
                if accelerator.is_main_process:
                    with open(ema_history_path, "a", encoding="utf-8") as f:
                        f.write(f"{global_step}\t{ema_avg:.6f}\t{ema_std:.6f}\n")
            else:
                anchor_results = utilities.sample_anchor(
                    args, transformer, vae, encoder_hidden_states, pooled_prompt_embeds,
                    guidance, sigma_scheduler, image_processor)
                group_results = utilities.sample_group(
                    args, transformer, vae, encoder_hidden_states, pooled_prompt_embeds,
                    guidance, sigma_scheduler, image_processor, anchor_results)

            images = anchor_results[3] + [img for key in group_results
                                          for img in group_results[key][1]]
            rewards = reward_fn(images, captions_batch)

            for stepidx in args.sde_stepidx:
                loss = method.compute_loss(
                    args, transformer, anchor_results, group_results, rewards,
                    encoder_hidden_states, pooled_prompt_embeds, guidance,
                    sigma_scheduler, stepidx,
                    ema_mean=ema_avg,
                    ema_std=(max(ema_var, 1e-8) ** 0.5) if ema_var is not None else None)
                accelerator.backward(loss / len(args.sde_stepidx))

            if accelerator.sync_gradients:
                total_norm_value = accelerator.clip_grad_norm_(
                    transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                if accelerator.is_main_process and global_step % 5 == 0:
                    anchor_results[3][0].save(
                        os.path.join(args.output_dir, f"step_{global_step}.jpg"))

                main_write(f"reward: {args.reward_name} score: {rewards.mean().item():.4f} ")
                main_write(f"global_step={global_step} ")
                main_write(f"grad_norm: {total_norm_value:.6f} ")
                main_write(f"lr={lr_scheduler.get_last_lr()[0]:.2e}\n")
                global_step += 1

            # ---------------- Saving ----------------
            if global_step > 0 and global_step % args.save_interval == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(transformer)
                    model_save_dir = os.path.join(args.output_dir, f"step_{global_step}")
                    if args.train_mode == "full":
                        full_state_dict = {k: v.to("cpu") for k, v in unwrapped_model.state_dict().items()}
                        unwrapped_model.save_pretrained(
                            model_save_dir, state_dict=full_state_dict,
                            safe_serialization=True)
                        del full_state_dict
                    else:
                        unwrapped_model.save_pretrained(model_save_dir)
                    main_print(f"Saved {'LoRA adapters' if args.train_mode == 'lora' else 'Full Model Weights'} "
                               f"to {model_save_dir}")
                    del unwrapped_model
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="PaveGRPO",
                        choices=["PaveGRPO", "AdaGRPO"])
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default=os.path.join(PRETRAINED_DIR, "FLUX"))
    parser.add_argument("--prompt_txt_path", type=str,
                        default=os.path.join(DATA_DIR, "prompts.txt"),
                        help="Plain-text prompt file, one prompt per line (encoded on the fly).")
    parser.add_argument("--encode_cache_size", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="./outputs/grpo")
    parser.add_argument("--reward_name", type=str, default="hpsv3", choices=["hpsv3", "hpsv2"])
    parser.add_argument("--hpsv2_clip_path", type=str,
                        default=os.path.join(PRETRAINED_DIR, "HPSv2", "open_clip_pytorch_model.bin"))
    parser.add_argument("--hpsv2_checkpoint", type=str,
                        default=os.path.join(PRETRAINED_DIR, "HPSv2", "HPS_v2.1_compressed.pt"))
    parser.add_argument("--train_mode", type=str, default="lora", choices=["lora", "full"])
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--denoising_steps", type=int, default=8)
    parser.add_argument("--sde_stepidx", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--sde_eta", type=float, default=0.8)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--group_size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adv_clip_max", type=float, default=5.0)
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--num_anneal_steps", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=300)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--seg_list", type=int, nargs="+", default=None)
    parser.add_argument("--ema_alpha", type=float, default=0.8)

    args = parser.parse_args()
    main(args)
