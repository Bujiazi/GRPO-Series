import bisect

import torch
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def assert_valid_sequence(data, K):
    if not data:
        return
    assert 0 <= data[0] and data[-1] < K, \
        f"elements must be within [0, {K}), got: [{data[0]}, {data[-1]}]"
    for i in range(len(data) - 1):
        assert data[i] < data[i + 1], \
            f"array must be strictly increasing and unique, found: {data[i]} -> {data[i + 1]}"


def get_rank(arr, target):
    return bisect.bisect_right(arr, target) - 1


def get_warmup_constant_anneal_scheduler(optimizer, num_warmup_steps, num_anneal_steps, num_training_steps):
    assert num_training_steps > num_warmup_steps + num_anneal_steps, \
        f"num_training_steps ({num_training_steps}) must exceed warmup({num_warmup_steps}) + anneal({num_anneal_steps})."

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > (num_training_steps - num_anneal_steps):
            remaining_steps = num_training_steps - current_step
            return max(0.0, float(remaining_steps) / float(max(1, num_anneal_steps)))
        else:
            return 1.0

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# FLUX latent utilities
# ---------------------------------------------------------------------------
def pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
    return latents


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    return latent_image_ids.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Advantage estimation
# ---------------------------------------------------------------------------
def compute_relative_adv(rewards, adv_clip_max):
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    return adv.clamp(-adv_clip_max, adv_clip_max)


# ---------------------------------------------------------------------------
# FlowSDE
# ---------------------------------------------------------------------------
def flowsde_std(sigma, sigma_fallback, eta):
    return torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_fallback, sigma))) * eta


def flowsde_prob_func(image_latents, pred_velocity, sigma_input, sigma_target, eta):
    std_dev_t = flowsde_std(sigma_input, sigma_target, eta)
    dt = sigma_target - sigma_input
    mean_next = (image_latents.float() * (1 + std_dev_t ** 2 / (2 * sigma_input) * dt)
                 + pred_velocity.float() * (1 + std_dev_t ** 2 * (1 - sigma_input) / (2 * sigma_input)) * dt)
    denom = 2 * (std_dev_t ** 2) * dt.abs()

    def log_prob_fn(latents, mean_next_latents):
        return -((latents.detach().float() - mean_next_latents) ** 2).mean(dim=(1, 2)) / denom

    return log_prob_fn, mean_next


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_anchor(args, transformer, vae, encoder_hidden_states, pooled_prompt_embeds,
                  guidance, sigma_scheduler, image_processor):
    device = encoder_hidden_states.device
    dtype = encoder_hidden_states.dtype
    B = encoder_hidden_states.shape[0]

    initial_noise = torch.randn((B, 16, args.height // 8, args.width // 8),
                                device=device, dtype=torch.bfloat16)
    image_latents = pack_latents(initial_noise, B, 16, args.height // 8, args.width // 8)
    latent_image_ids = prepare_latent_image_ids(B, args.height // 16, args.width // 16, device, dtype)
    text_ids = torch.zeros(encoder_hidden_states.shape[1], 3, device=device, dtype=dtype)

    all_latents = [image_latents.cpu()]
    all_pred_noise, all_pred_clean = [], []
    for sigma_idx in range(len(sigma_scheduler) - 1):
        t = sigma_scheduler[sigma_idx]
        timestep = t.expand(image_latents.shape[0]).to(image_latents.dtype)
        pred_velocity = transformer(
            hidden_states=image_latents,
            timestep=timestep,
            guidance=guidance.expand(image_latents.shape[0]),
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]
        pred_noise = image_latents.float() + (1 - sigma_scheduler[sigma_idx]) * pred_velocity
        pred_clean = image_latents.float() - sigma_scheduler[sigma_idx] * pred_velocity
        image_latents = image_latents.float() - (sigma_scheduler[sigma_idx] - sigma_scheduler[sigma_idx + 1]) * pred_velocity
        image_latents = image_latents.to(dtype=dtype)
        all_latents.append(image_latents.cpu())
        all_pred_noise.append(pred_noise.cpu())
        all_pred_clean.append(pred_clean.cpu())

    image_latents = unpack_latents(image_latents, args.height, args.width, 8)
    image_latents = (image_latents / vae.config.scaling_factor) + vae.config.shift_factor
    image_pixel = vae.decode(image_latents, return_dict=False)[0]
    image_pixel = image_processor.postprocess(image_pixel, output_type="pil")
    return [all_latents, all_pred_noise, all_pred_clean, image_pixel, latent_image_ids]


@torch.no_grad()
def sample_group(args, transformer, vae, encoder_hidden_states, pooled_prompt_embeds,
                 guidance, sigma_scheduler, image_processor, anchor_results):

    device = encoder_hidden_states.device
    dtype = encoder_hidden_states.dtype
    G = args.group_size
    text_ids = torch.zeros(encoder_hidden_states.shape[1], 3, device=device, dtype=dtype)
    ehs_group = encoder_hidden_states.expand(G, *(-1,) * (encoder_hidden_states.dim() - 1))
    ppe_group = pooled_prompt_embeds.expand(G, *(-1,) * (pooled_prompt_embeds.dim() - 1))

    results = {}
    for stepidx in args.sde_stepidx:
        pred_noise = anchor_results[1][stepidx].to(device)
        pred_clean = anchor_results[2][stepidx].to(device)
        expand_noise = torch.randn((G,) + pred_noise.shape[1:], device=device, dtype=pred_noise.dtype)

        sigma = sigma_scheduler[stepidx]
        std_dev_t = flowsde_std(sigma, sigma_scheduler[1], args.sde_eta)
        latents = sigma * pred_noise + (1 - sigma) * pred_clean
        pred_velocity = pred_noise - pred_clean
        dt = sigma_scheduler[stepidx + 1] - sigma
        prev_mean = (latents * (1 + std_dev_t ** 2 / (2 * sigma) * dt)
                     + pred_velocity * (1 + std_dev_t ** 2 * (1 - sigma) / (2 * sigma)) * dt)
        image_latents = prev_mean.expand(G, *(-1,) * (pred_clean.dim() - 1)) \
            + std_dev_t * torch.sqrt(-dt) * expand_noise
        image_latents = image_latents.to(dtype)
        sde_latents = image_latents.cpu()

        for sigma_idx in range(stepidx + 1, len(sigma_scheduler) - 1):
            t = sigma_scheduler[sigma_idx]
            timestep = t.expand(image_latents.shape[0]).to(image_latents.dtype)
            pred_velocity = transformer(
                hidden_states=image_latents,
                timestep=timestep,
                guidance=guidance.expand(image_latents.shape[0]),
                pooled_projections=ppe_group,
                encoder_hidden_states=ehs_group,
                txt_ids=text_ids,
                img_ids=anchor_results[-1],
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
            image_latents = image_latents.float() - (sigma_scheduler[sigma_idx] - sigma_scheduler[sigma_idx + 1]) * pred_velocity
            image_latents = image_latents.to(dtype)

        image_latents = unpack_latents(image_latents, args.height, args.width, 8)
        image_latents = (image_latents / vae.config.scaling_factor) + vae.config.shift_factor
        image_pixel = vae.decode(image_latents, return_dict=False)[0]
        image_pixel = image_processor.postprocess(image_pixel, output_type="pil")
        results[str(stepidx)] = [sde_latents, image_pixel]

    return results
