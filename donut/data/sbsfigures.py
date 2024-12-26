import json, os
import random
from typing import Any, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import DonutProcessor

added_tokens = []

class OursDataset(Dataset):
    
    def __init__(
        self,
        dataset: str,
        max_length: int,
        processor : DonutProcessor = None,
        split: str = "train",
        ignore_id: int = -100,
        prompt_end_token: str = None,
        task_prefix: str = '<chartqa>',
        sort_json_key: bool = True,
    ):
        super().__init__()
        self.dataset = dataset
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.prompt_end_token = prompt_end_token 
        self.sort_json_key = sort_json_key
        self.processor = processor
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        self.task_prefix = task_prefix

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        sample = self.dataset[idx]

        img = sample['image']  
        if not isinstance(img, Image.Image): 
            img = Image.fromarray(img)

        if img.mode != "RGB":
            img = img.convert("RGB")

        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()

        query = sample['query']
        label = sample['label']
        processed_parse = (
            self.task_prefix + " " + query + " " + self.prompt_end_token + " " + label + self.processor.tokenizer.eos_token
        )

        input_ids = self.processor.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.processor.tokenizer.pad_token_id
            ] = self.ignore_id  
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  
            return pixel_values, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return pixel_values, input_ids, prompt_end_index, processed_parse