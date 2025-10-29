from datasets import load_dataset
from transformers import AutoTokenizer
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
import numpy as np
import torch
import os
import datasets
from tqdm import tqdm
import dotenv
dotenv.load_dotenv()

#tokenizer = AutoTokenizer.from_pretrained("nvidia/Nemotron-4-340B-Base")
#tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")


output = "./data"
os.makedirs(output, exist_ok=True)

# 40M de samples
for sample in ["fineweb-edu-dedup"]:
    ds = load_dataset("HuggingFaceTB/smollm-corpus", sample, split="train", num_proc=32)
    # select 40_000_000 samples
    ds = ds.select(range(40_000_000))
    ds = ds.map(lambda x: {'text': f'<|im_start|>{x["text"]}<|endoftext|>'})
    ds = ds.map(lambda batch: tokenizer(batch["text"], truncation=False, padding=False), num_proc=32, batched=True)
    print(ds)
    
    prefix = f"{output}/{sample}"
    builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
    for i, sample in enumerate(tqdm(ds)):
        builder.add_item(torch.tensor(sample['input_ids'], dtype=torch.int32))
        builder.end_document()
            
    builder.finalize(f"{prefix}.idx")
    print("IndexedDataset salvo em:", prefix)


