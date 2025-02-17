
from sys import exit
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset, DatasetDict
import pandas as pd
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import DataCollatorWithPadding
from typing import Union, Optional
from transformers.tokenization_utils_base import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from dataclasses import dataclass
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, LlamaTokenizer
from transformers import LlamaConfig
from transformers import LlamaForCausalLM
from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase
from pathlib import Path
from env import env_is_debug
# %%
words = open('data/codes_mots/civil_mots.txt', 'r').read().splitlines()
nb_words = len(words)
print("nb_words =", nb_words)

#%% [markdown]
# ## Préparation des données
# Construction d'un jeu de données à partir des mots du code civil et du code du travail


#%%
words2 = open('data/codes_mots/code_travail_mots.txt', 'r').read().splitlines()
print(f'nb_words2 = {len(words2)}')

df = pd.DataFrame({'text': words + words2, 'source' : ['civil']*len(words) + ['travail']*len(words2)})
# %%
chars = sorted(list(set(''.join(words))))
nb_chars = len(chars) + 1  # On ajoute 1 pour EOS

#%% 

tokenizer_name = "keeeeenw/MicroLlama"
tokenizer : LlamaTokenizer= LlamaTokenizer.from_pretrained(tokenizer_name, add_eos_token = True)
tokenizer.add_special_tokens({"pad_token": "<pad>"})


#%% 

df = df.sample(frac=1).reset_index(drop=True)  # Corrected shuffle method
df["is_train"] = df.index % 5 != 0
# dataset = ds.dataset(pa.Table.from_pandas(df).to_batches())

train_df = df[df["is_train"]]  # 80% for training
valid_df = df[~df["is_train"]]  # 20% for validation

# Convert DataFrames to Hugging Face Dataset objects
train_dataset = Dataset(pa.Table.from_pandas(train_df))
valid_dataset = Dataset(pa.Table.from_pandas(valid_df))

# Combine into a DatasetDict
hg_dataset = DatasetDict({
    "train": train_dataset,
    "validation": valid_dataset
})
# reducing the size of the dataset
if env_is_debug():
    import random 
    random_100 = random.sample(range(0, len(hg_dataset['train'])), 100)
    random_100_val = random.sample(range(0, len(hg_dataset['validation'])), 100)
    print(f"Reducing dataset size for debugging")
    hg_dataset['train'] = hg_dataset['train'].select(random_100)
    hg_dataset['validation'] = hg_dataset['validation'].select(random_100_val)



small_config = {
    "num_hidden_layers": 3,
    "hidden_size": 64,
    "intermediate_size": 256,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    # Required additional parameters
    "max_position_embeddings": 512,  # Maximum sequence length
    "vocab_size": 32000,            # Size of the tokenizer vocabulary
    "rms_norm_eps": 1e-6,          # Layer normalization epsilon
    "bos_token_id": 1,             # Beginning of sequence token
    "eos_token_id": 2,             # End of sequence token
    "pad_token_id": 0,             # Padding token
}

smallconfig = LlamaConfig(**small_config)
smallmodel = LlamaForCausalLM(smallconfig)

Path('/.results').mkdir(exist_ok = True)
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    use_cpu=False,
    report_to = None
)

@dataclass
class DataCollatorCustom:
    tokenizer: PreTrainedTokenizerBase
    return_tensors: str = "pt"
    def __call__(self, features:list[dict]):
        input_ids = []
        attention_masks = []
        labels = []
        for sample in features:
            input_ids.append(sample["input_ids"])
            attention_masks.append(sample["attention_mask"])
            if "labels" in sample:
                labels.append(sample["labels"])
            else:
                labels.append(sample["input_ids"].copy())
        longest_input = max([len(input_id) for input_id in input_ids])
        input_ids = [input_id + [self.tokenizer.pad_token_id]*(longest_input - len(input_id)) for input_id in input_ids]
        labels = [label + [-100]*(longest_input - len(label)) for label in labels]
        attention_masks = [attention_mask + [0]*(longest_input - len(attention_mask)) for attention_mask in attention_masks]
        input_ids_tensor = torch.tensor(input_ids)
        labels_tensor = torch.tensor(labels)
        attention_masks_tensor = torch.tensor(attention_masks)
        # labels_tensor[labels_tensor == self.tokenizer.pad_token_id] = -100
        returndict = {"input_ids": input_ids_tensor, 
                "attention_mask": attention_masks_tensor,
                "labels": labels_tensor}
        print('saving tensors for batch 1')
        torch.save(input_ids_tensor, 'temp/input_ids.pt')
        torch.save(labels_tensor, 'temp/labels.pt')
        exit()
        return returndict
    
data_collator = DataCollatorCustom(tokenizer)

def tokenize_with_tokenizer(examples: dict, tokenizer: PreTrainedTokenizerBase):

    outputs = tokenizer(
        examples['text'],
        padding="longest",
        add_special_tokens=True
    )
    return outputs

hg_dataset_tokenized = hg_dataset.map(
    lambda examples: tokenize_with_tokenizer(examples, tokenizer),
    batched=False,
    num_proc=1
)

def MyCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    
    def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
        import torch.nn as nn
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        if reduction == "sum":
            loss = loss / num_items_in_batch
        return loss

    print('saving tensors for batch 1')
    torch.save(logits, 'temp/logits_end.pt')
    torch.save(labels, 'temp/labels_end.pt')
    exit()
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    
    # TO REMOVE 
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss


from typing import Callable
from transformers import LlamaForCausalLM
from transformers import Trainer

print('starting trainer')
trainer_v2 =  Trainer(
    model=smallmodel,
    args=training_args,
    data_collator=data_collator,
    train_dataset=hg_dataset_tokenized["train"],
    eval_dataset=hg_dataset_tokenized["validation"],
)
# trainer_v2.train()

# # %%


# # If you want to add some randomness, here's a version with temperature:
# #%%
# max_length = 50
# temperature = 0.7
# generated_tokens = ""
# input_ids = tokenizer(prompt, return_tensors="pt")['input_ids']
# generated_tokens = input_ids[0].tolist()

# for _ in range(max_length):
#     outputs = smallmodel(input_ids=torch.tensor([generated_tokens], dtype=torch.long))
#     next_token_logits = outputs.logits[:, -1, :] / temperature
    
#     # Apply softmax to get probabilities
#     probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
    
#     # Sample from the distribution
#     next_token = torch.multinomial(probs[0], num_samples=1).item()
    
#     generated_tokens.append(next_token)
    
#     if next_token == tokenizer.eos_token_id:
#         print("EOS token found, stopping generation")
#         break
# generated_tokens = [int(token) for i, token in enumerate(generated_tokens) if i >= len(input_ids)]
# tokenizer.decode(generated_tokens)  
# #%%
# # trying to understand the causal lm loss 
# from transformers import PreTrainedTokenizer
# from transformers.loss.loss_utils import 
# # lets add the input ids as well to compare to the labels
# # i want to see if the datacollator is used correctly
# logits = torch.load('temp/logits.pt')
# labels = torch.load('temp/labels.pt')
# print(f'shapes = {logits.shape}, {labels.shape}')
# loss = ForCausalLMLoss(labels, logits, tokenizer.vocab_size, 4)
# #%%

# print("Padding token:", tokenizer.pad_token)
# print("Padding token ID:", tokenizer.pad_token_id)
# # %%
# from importlib import reload
# import transformers.loss.loss_utils
# reload(transformers.loss.loss_utils)
# # %%
# import inspect
# import transformers.loss.loss_utils

# # Check source code of the loss function
# # print(inspect.getsource(transformers.loss.loss_utils.ForCausalLMLoss))

# # Check where it's loaded from
# # print(inspect.getfile(transformers.loss.loss_utils.ForCausalLMLoss))

# # Check if class has expected attributes/methods
# # print(inspect.getmembers(transformers.loss.loss_utils.ForCausalLMLoss))

# # %%
