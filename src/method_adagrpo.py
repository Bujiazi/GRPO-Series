import numpy as np
import torch

from utils.utilities import (compute_relative_adv, flowsde_prob_func, get_rank,
                              sample_anchor, sample_group)


@torch.no_grad()
def sample_evaluate_and_select_ema(args, transformer, vae, encoder_hidden_states,
                                   pooled_prompt_embeds, captions, guidance,
                                   sigma_scheduler, image_processor, reward_fn,
                                   ema_avg, ema_var):
    anchor_results = sample_anchor(args, transformer, vae, encoder_hidden_states,
                                   pooled_prompt_embeds, guidance, sigma_scheduler,
                                   image_processor)
    scores = reward_fn(anchor_results[3], captions).cpu().numpy()

    batch_mean, batch_var = float(scores.mean()), float(scores.var())
    if ema_avg is None:
        new_ema_avg, new_ema_var = batch_mean, batch_var
    else:
        a = args.ema_alpha
        new_ema_avg = a * ema_avg + (1 - a) * batch_mean
        new_ema_var = a * ema_var + (1 - a) * batch_var

    selected_idx = int(np.argmin(np.abs(scores - new_ema_avg)))
    sl = slice(selected_idx, selected_idx + 1)
    sel_anchor_results = [
        [t[sl] for t in anchor_results[0]],
        [t[sl] for t in anchor_results[1]],
        [t[sl] for t in anchor_results[2]],
        [anchor_results[3][selected_idx]],
        anchor_results[4],
    ]
    group_results = sample_group(args, transformer, vae,
                                 encoder_hidden_states[sl], pooled_prompt_embeds[sl],
                                 guidance, sigma_scheduler, image_processor,
                                 sel_anchor_results)
    return {
        "new_ema_avg": new_ema_avg,
        "new_ema_var": new_ema_var,
        "selected_idx": selected_idx,
        "all_scores": scores.tolist(),
        "anchor_results": sel_anchor_results,
        "group_results": group_results,
        "encoder_hidden_states": encoder_hidden_states[sl],
        "pooled_prompt_embeds": pooled_prompt_embeds[sl],
        "captions": [captions[selected_idx]],
    }


def compute_ema_adv(rewards, ema_mean, ema_std, adv_clip_max):
    z = (rewards - ema_mean) / (ema_std + 1e-8)

    pos_mask, neg_mask = z > 0, z < 0
    sum_pos = z[pos_mask].sum()
    sum_neg = z[neg_mask].abs().sum()
    if sum_pos < 1e-6 or sum_neg < 1e-6:
        z = z - z.mean()
    else:
        z[pos_mask] /= sum_pos
        z[neg_mask] /= sum_neg
    return z.clamp(-adv_clip_max, adv_clip_max)


def compute_loss(args, transformer, anchor_results, group_results, rewards,
                 encoder_hidden_states, pooled_prompt_embeds, guidance,
                 sigma_scheduler, stepidx, ema_mean=None, ema_std=None):
    device = encoder_hidden_states.device
    current_idx = get_rank(args.sde_stepidx, stepidx)
    rewards_group = rewards[args.group_size * current_idx + 1:
                            args.group_size * (current_idx + 1) + 1].to(device)
    latents_group = group_results[str(stepidx)][0].to(device)

    image_latents = anchor_results[0][stepidx].to(device)
    t = sigma_scheduler[stepidx]
    pred_velocity = transformer(
        hidden_states=image_latents.detach(),
        timestep=t.expand(image_latents.shape[0]).to(image_latents.dtype),
        guidance=guidance.expand(image_latents.shape[0]),
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=encoder_hidden_states,
        txt_ids=torch.zeros(encoder_hidden_states.shape[1], 3,
                            device=device, dtype=image_latents.dtype),
        img_ids=anchor_results[-1],
        joint_attention_kwargs={},
        return_dict=False,
    )[0]
    log_prob_fn, mean_next = flowsde_prob_func(
        image_latents, pred_velocity, t, sigma_scheduler[stepidx + 1], args.sde_eta)

    adv = compute_relative_adv(rewards_group, args.adv_clip_max) \
        + compute_ema_adv(rewards_group, ema_mean, ema_std, args.adv_clip_max)

    log_prob = log_prob_fn(latents_group, mean_next)
    ratio = torch.exp(log_prob - log_prob.detach())
    return (-adv.detach().float() * ratio).mean()
