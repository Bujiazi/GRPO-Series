import torch

from utils.utilities import compute_relative_adv, flowsde_prob_func, get_rank


def _paved_transitions(image_latents, transformer, sigma_init, sigma_end, seg_list,
                       encoder_hidden_states, pooled_prompt_embeds, guidance,
                       latent_image_ids, eta):
    seg_list = seg_list or []
    assert all(n > 1 for n in seg_list), "seg_list elements must be > 1"
    assert len(seg_list) == len(set(seg_list)), "seg_list elements must be unique"
    text_ids = torch.zeros(encoder_hidden_states.shape[1], 3,
                           device=image_latents.device, dtype=image_latents.dtype)

    def _forward_batch(latents_list, sigma_list):
        B = latents_list[0].shape[0]
        latents_batch = torch.cat(latents_list, dim=0)
        sigma_batch = torch.cat([s.expand(B) for s in sigma_list], dim=0).to(latents_batch.dtype)
        v = transformer(
            hidden_states=latents_batch,
            timestep=sigma_batch,
            guidance=guidance.expand(latents_batch.shape[0]),
            pooled_projections=pooled_prompt_embeds.expand(
                latents_batch.shape[0], *(-1,) * (pooled_prompt_embeds.dim() - 1)),
            encoder_hidden_states=encoder_hidden_states.expand(
                latents_batch.shape[0], *(-1,) * (encoder_hidden_states.dim() - 1)),
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]
        return torch.split(v, B, dim=0)

    results = []
    v1 = _forward_batch([image_latents], [sigma_init])[0]
    results.append(flowsde_prob_func(image_latents, v1, sigma_init, sigma_end, eta))
    if not seg_list:
        return results

    states = {}
    for n in seg_list:
        dt = (sigma_end - sigma_init) / n
        states[n] = (image_latents + dt * v1, sigma_init + dt)

    seg_results = {}
    for step in range(2, max(seg_list) + 1):
        active = [n for n in seg_list if n in states]
        if not active:
            break
        v_list = _forward_batch([states[n][0] for n in active],
                                [states[n][1] for n in active])
        for n, v_n in zip(active, v_list):
            cur_latents, cur_sigma = states[n]
            if step == n:
                seg_results[n] = flowsde_prob_func(cur_latents, v_n, cur_sigma, sigma_end, eta)
                del states[n]
            else:
                dt = (sigma_end - sigma_init) / n
                states[n] = (cur_latents + dt * v_n, cur_sigma + dt)

    for n in seg_list:
        results.append(seg_results[n])
    return results


def compute_loss(args, transformer, anchor_results, group_results, rewards,
                 encoder_hidden_states, pooled_prompt_embeds, guidance,
                 sigma_scheduler, stepidx, ema_mean=None, ema_std=None):
    device = encoder_hidden_states.device
    current_idx = get_rank(args.sde_stepidx, stepidx)
    rewards_group = rewards[args.group_size * current_idx + 1:
                            args.group_size * (current_idx + 1) + 1].to(device)
    latents_group = group_results[str(stepidx)][0].to(device)
    adv = compute_relative_adv(rewards_group, args.adv_clip_max)

    image_latents = anchor_results[0][stepidx].to(device)
    transitions = _paved_transitions(
        image_latents.detach(), transformer,
        sigma_scheduler[stepidx], sigma_scheduler[stepidx + 1], args.seg_list,
        encoder_hidden_states, pooled_prompt_embeds, guidance,
        anchor_results[-1], args.sde_eta)

    loss = 0.0
    for idx, (log_prob_fn, mean_next) in enumerate(transitions):
        log_prob = log_prob_fn(latents_group, mean_next)
        ratio = torch.exp(log_prob - log_prob.detach())
        loss_grpo = -adv.detach().float() * ratio
        weight = 1.0 if idx == 0 else 1.0 / len(args.seg_list)
        loss = loss + weight * loss_grpo.mean()
    return loss
