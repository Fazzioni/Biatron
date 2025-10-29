from transformers import AutoTokenizer
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
import numpy as np
import torch
import os
import datasets
from tqdm import tqdm
import dotenv
dotenv.load_dotenv()

tokenizer = AutoTokenizer.from_pretrained("Fazzioni/tokenizer-fineweb.50M-32k")

output = "./resoaning"

os.makedirs(output, exist_ok=True)

data = datasets.load_from_disk("/data/reasoning-v1-20m-emb_tokenized_ordered")

data = data.map(lambda x: {'text': f"{x['prompt']} \n\n {x['thought']} \n\n {x['answer']}"})
data = data.map(lambda batch: tokenizer(batch["text"], truncation=False, padding=False), num_proc=64, batched=True)

#prefix = f"{output}/reasoning-v1"
#builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
#for i, sample in enumerate(tqdm(data)):
#    builder.add_item(torch.tensor(sample['input_ids'], dtype=torch.int32))
#    builder.end_document()
#        
#builder.finalize(f"{prefix}.idx")
#print("IndexedDataset salvo em:", prefix)



def write_chunk(args):
    """Função chamada em paralelo para escrever um subconjunto do dataset."""
    chunk, chunk_id, prefix = args
    chunk_prefix = f"{prefix}_part{chunk_id:03d}"
    builder = IndexedDatasetBuilder(f"{chunk_prefix}.bin", dtype=np.int32)

    for sample in chunk:
        builder.add_item(torch.tensor(sample["input_ids"], dtype=torch.int32))
        builder.end_document()

    builder.finalize(f"{chunk_prefix}.idx")
    return chunk_prefix

# =========================
# Escrita paralela
# =========================

prefix = os.path.join("reasoning-v1")
num_workers = 48
chunk_size = 500_000  # ajuste conforme o tamanho do dataset

chunks = [
    (data[i : i + chunk_size], idx // chunk_size, prefix)
    for idx, i in enumerate(range(0, len(data), chunk_size))
]

print(f"Escrevendo {len(chunks)} chunks com {num_workers} processos...")

from multiprocessing import Pool, cpu_count

with Pool(num_workers) as pool:
    results = list(tqdm(pool.imap_unordered(write_chunk, chunks), total=len(chunks)))

print("Chunks salvos em:")
for path in sorted(results):
    print(" -", path)

print("\n✅ Escrita paralela concluída!")