from collections import OrderedDict

import torch
from torch.utils.data import Dataset


class CaptionDataset(Dataset):

    def __init__(self, captions):
        self.captions = list(captions)

    def __getitem__(self, idx):
        return idx, self.captions[idx]

    def __len__(self):
        return len(self.captions)


def caption_collate_function(batch):
    return {"indices": [b[0] for b in batch], "captions": [b[1] for b in batch]}


def load_captions(prompt_txt_path):
    with open(prompt_txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class PromptEncoder:

    def __init__(self, pretrained_path, device, dtype=torch.bfloat16, cache_size=2048):
        from transformers import (CLIPTextModel, CLIPTokenizer,
                                  T5EncoderModel, T5TokenizerFast)
        self.device = device
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_path, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            pretrained_path, subfolder="text_encoder_2", torch_dtype=dtype).to(device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer")
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(pretrained_path, subfolder="tokenizer_2")

    @torch.no_grad()
    def encode(self, captions):
        pe_list, ppe_list = [], []
        for cap in captions:
            cached = self.cache.get(cap)
            if cached is None:
                # T5 -> prompt_embeds [1, 256, 4096]
                t5_in = self.tokenizer_2(cap, padding="max_length", max_length=256,
                                         truncation=True, return_tensors="pt")
                pe = self.text_encoder_2(t5_in.input_ids.to(self.device))[0].cpu()
                # CLIP -> pooled_prompt_embeds [1, 768]
                clip_in = self.tokenizer(cap, padding="max_length", max_length=77,
                                         truncation=True, return_tensors="pt")
                ppe = self.text_encoder(clip_in.input_ids.to(self.device)).pooler_output.cpu()
                if len(self.cache) >= self.cache_size:
                    self.cache.popitem(last=False)
                self.cache[cap] = (pe, ppe)
            else:
                self.cache.move_to_end(cap)
                pe, ppe = cached
            pe_list.append(pe)
            ppe_list.append(ppe)
        return (torch.cat(pe_list, dim=0).to(self.device),
                torch.cat(ppe_list, dim=0).to(self.device))
