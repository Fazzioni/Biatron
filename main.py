import torch
from transformers import  AutoTokenizer, HfArgumentParser  
from biatron import BiatronForCausalLM
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from dataclasses import dataclass, field
import dotenv
dotenv.load_dotenv()

@dataclass
class DatasetConfig:
    dataset_name : str = field(default=None, metadata={"help": "The name of the dataset to use."})

def train(training_args:SFTConfig, dataset_args:DatasetConfig):
    
    tokenizer = AutoTokenizer.from_pretrained("Fazzioni/biatron-345m-it")
    model = BiatronForCausalLM.from_pretrained("Fazzioni/biatron-345m", 
                                               revision='checkpoint-122100', 
                                               attn_implementation='sdpa',
                                               use_cache=False if training_args.gradient_checkpointing else True,
                                               torch_dtype=torch.bfloat16
                                               )
    model.add_new_tokens(len(tokenizer))

    dataset = load_dataset(dataset_args.dataset_name, split='train')
    dataset = dataset.shuffle(seed=0)
    dataset = dataset.filter(lambda s: s['clarity_score'] >= 7)
    dataset = dataset.filter(lambda s: s['relevance_score'] >= 7)
    dataset = dataset.filter(lambda s: s['accuracy_score'] >= 7)
    dataset = dataset.filter(lambda s: s['completeness_score'] >= 7)
    dataset = dataset.filter(lambda s: s['educational'] >= 7)
    
    
    dataset = dataset.train_test_split(test_size=0.1, seed=0)
    train_dataset , eval_dataset = dataset['train'] , dataset['test']
    print(train_dataset)
    
    # filter dataset dont have messages
    
    
    #def data_collator(features: list) -> dict:
    #    batch = tokenizer(features['messages'],
    #        padding="longest",
    #        truncation=True,
    #        max_length=training_args.max_length,
    #        return_tensors="pt",
    #    )
    #    batch["labels"] = batch["input_ids"].clone()
    #    return batch

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)    

if __name__ == "__main__":
    parser = HfArgumentParser((SFTConfig, DatasetConfig))
    training_args, dataset_args = parser.parse_args_into_dataclasses()
    train(training_args, dataset_args)