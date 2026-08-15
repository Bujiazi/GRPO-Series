import os
from collections.abc import Mapping

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "pretrained_models")


def _match_captions(captions, n):
    if isinstance(captions, str):
        return [captions] * n
    captions = list(captions)
    if len(captions) == 1 and n > 1:
        return captions * n
    assert len(captions) == n, f"len(captions)={len(captions)} != len(images)={n}"
    return captions


class HPSv3RewardInferencer:

    def __init__(self, config_path=None, checkpoint_path=None, device='cuda', differentiable=False):
        from hpsv3.dataset.utils import process_vision_info
        from hpsv3.dataset.data_collator_qwen import (
            prompt_with_special_token, prompt_without_special_token, INSTRUCTION)
        from hpsv3.utils.parser import (
            ModelConfig, PEFTLoraConfig, TrainingConfig, DataConfig, parse_args_with_yaml)
        from hpsv3.train import create_model_and_processor

        self._process_vision_info = process_vision_info
        self._prompt_with_special_token = prompt_with_special_token
        self._prompt_without_special_token = prompt_without_special_token
        self._INSTRUCTION = INSTRUCTION

        config_path = config_path or os.path.join(PRETRAINED_DIR, "HPSv3", "HPSv3_7B.yaml")
        checkpoint_path = checkpoint_path or os.path.join(PRETRAINED_DIR, "HPSv3", "HPSv3.safetensors")

        (data_config, training_args, model_config, peft_lora_config), config_path = (
            parse_args_with_yaml(
                (DataConfig, TrainingConfig, ModelConfig, PEFTLoraConfig), config_path, is_train=False
            )
        )
        model_config.model_name_or_path = os.path.join(
            PRETRAINED_DIR, "HPSv3", "Qwen2-VL-7B-Instruct")
        training_args.output_dir = os.path.join(
            training_args.output_dir, config_path.split("/")[-1].split(".")[0]
        )
        model, processor, peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
            differentiable=differentiable,
        )

        self.device = device
        self.use_special_tokens = model_config.use_special_tokens

        if checkpoint_path.endswith('.safetensors'):
            import safetensors.torch
            state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        if "model" in state_dict:
            state_dict = state_dict["model"]

        if any('model.language_model' in key for key in model.state_dict().keys()) and not any('language_model' in key for key in state_dict.keys()):
            updated_state_dict = {}
            for key, value in state_dict.items():
                if 'visual' in key:
                    updated_state_dict[key.replace('visual', 'model.visual')] = value
                elif 'model' in key:
                    updated_state_dict[key.replace('model', 'model.language_model')] = value
                else:
                    updated_state_dict[key] = value
            state_dict = updated_state_dict

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self.model = model
        self.processor = processor

        self.model.to(self.device)
        self.data_config = data_config

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side='right'):
        """Pad the sequences to the maximum length."""
        assert padding_side in ['right', 'left']
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask

        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == 'right' else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(sequences, padding, 'constant', self.processor.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, 'constant', 0)

        return sequences_padded, attention_mask_padded

    def _prepare_input(self, data):
        """Prepare `inputs` before feeding them to the model."""
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs):
        """Prepare `inputs` before feeding them to the model."""
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs

    def prepare_batch(self, image_paths, prompts):
        max_pixels = 256 * 28 * 28
        min_pixels = 256 * 28 * 28
        message_list = []
        for text, image in zip(prompts, image_paths):
            out_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": max_pixels,
                            "max_pixels": max_pixels,
                        },
                        {
                            "type": "text",
                            "text": (
                                self._INSTRUCTION.format(text_prompt=text)
                                + self._prompt_with_special_token
                                if self.use_special_tokens
                                else self._prompt_without_special_token
                            ),
                        },
                    ],
                }
            ]

            message_list.append(out_message)

        image_inputs, _ = self._process_vision_info(message_list)

        batch = self.processor(
            text=self.processor.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch

    @torch.inference_mode()
    def reward(self, image_paths, prompts):
        batch = self.prepare_batch(image_paths, prompts)
        rewards = self.model(
            return_dict=True,
            **batch
        )["logits"]

        return rewards


def build_reward_fn(args, device):
    if args.reward_name == "hpsv3":
        reward_model = HPSv3RewardInferencer(device=device)

        @torch.no_grad()
        def reward_fn(images, captions):
            with torch.amp.autocast("cuda"):
                scores = reward_model.reward(images, _match_captions(captions, len(images)))
            return scores[:, 0].float()

    elif args.reward_name == "hpsv2":
        from hpsv2.src.open_clip import create_model_and_transforms
        from hpsv2.src.open_clip import get_tokenizer as get_tokenizer_hpsv2
        model, _, processor_val = create_model_and_transforms(
            "ViT-H-14",
            args.hpsv2_clip_path,
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )
        checkpoint = torch.load(args.hpsv2_checkpoint, map_location="cuda")
        model.load_state_dict(checkpoint["state_dict"])
        del checkpoint
        reward_model = model.to(device).eval()
        tokenizer = get_tokenizer_hpsv2("ViT-H-14")

        @torch.no_grad()
        def reward_fn(images, captions):
            captions = _match_captions(captions, len(images))
            scores = []
            with torch.amp.autocast("cuda"):
                for image_pil, cap in zip(images, captions):
                    image = processor_val(image_pil).unsqueeze(0).to(device)
                    text = tokenizer([cap]).to(device)
                    outputs = reward_model(image, text)
                    logits_per_image = outputs["image_features"] @ outputs["text_features"].T
                    scores.append(torch.diagonal(logits_per_image).float())
            return torch.stack(scores).flatten()

    else:
        raise ValueError(f"Unknown reward_name: {args.reward_name} (supported: hpsv3, hpsv2)")

    return reward_fn
