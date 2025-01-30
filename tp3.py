
# %% [markdown]
# ## Données sources

# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
from datasets import Dataset, DatasetDict
import pandas as pd

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
df.sample(5)
# %%
chars = sorted(list(set(''.join(words))))
print(chars)
nb_chars = len(chars) + 1  # On ajoute 1 pour EOS
print("nb_chars =", nb_chars)
# Fun fact: il n'y a pas de 'k' dans le code civil

# %%
# Dictionnaire permettant de passer d'un caractère à son identifiant entier
ctoi = {c:i+1 for i,c in enumerate(chars)}
ctoi['.'] = 0
print("CTOI =", ctoi)
# Dictionnaire permettant permettant de passer d'un entier à son caractère
itoc = {i:s for s,i in ctoi.items()}
print("ITOC =", itoc)
# '.' a l'indice 0

# %% [markdown]
# ## Telechargement d'un tokenizer 
#%% 

from transformers import PreTrainedTokenizer, LlamaTokenizer
tokenizer_name = "keeeeenw/MicroLlama"
tokenizer : LlamaTokenizer= LlamaTokenizer.from_pretrained(tokenizer_name)
type(tokenizer)
# a tokenizer encodes a sentence into a list of tokens
tokenizer.encode("tanguy")
tokenizer.decode([1,260, 2375])
tokenizer.get_vocab().__len__()
tokenizer.get_vocab()["angu"]
tokenizer.special_tokens_map
tokenizer
# a tokenizer is essentially a trained dictionary 
# maybe i can retrain on code civil data 

#%% 
# build dataset function
tokenizer("hello there", "my name is tanguy")
tokenizer.decode([1, 22172, 727, 1, 590, 1024, 338, 260, 2375, 29891])

# %% [markdown]
tokenizer.pad_token = tokenizer.eos_token
def tokenize(examples):
    return tokenizer(examples['text'], padding="longest", truncation=True, max_length=512)

# %%

word = words[0]
tokenized = tokenizer(word,  padding='longest', truncation=True, max_length=512)
print(tokenized)

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

#%% 
hg_dataset_tokenized = hg_dataset.map(tokenize)

#%%

inputid = hg_dataset_tokenized["train"][10]['input_ids']
inputid
print(len(inputid), type(inputid))
for sample in hg_dataset_tokenized["train"]:
    if len(sample['input_ids']) > 16:
        print(sample)
        break


#%% 
# how does padding work for a language model 
# transformer input shape is  (B,T) long -> B,T,C -> B,T,vocab_size
# so T varies? 
# when we create a batch, we must pad to the longest sample in that batch 

#%%
# how does the input size of the sequence affect the training ? 
# can any model take in any max sequence length ?

from transformers import AutoModelForCausalLM
model_name = tokenizer_name
print(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
#%% 

model.model.layers = model.model.layers[:2]
input_ids = torch.tensor([inputid])
# labels = input_ids.clone()
input_ids_with_padding = torch.tensor([inputid + [tokenizer.pad_token_id]*5 + inputid])
labels = input_ids_with_padding.clone()
attention_mask = torch.tensor([[1]*len(inputid) + [0]*5 + [1]*len(inputid)])
#%%
"""
In the code, the attention mask will be transformed to this matrix
print(input_ids_with_padding.shape)
array([[  0., -inf, -inf],
       [  0.,   0., -inf],
       [  0.,   0.,   0.]], dtype=float32
"""
def graph_attention_mask():
    import numpy as np
    import matplotlib.pyplot as plt
    mymask = attention_mask.numpy()[0,0].copy()
    mymask[mymask < -1e10] = 1
    mymask = 1-mymask
    plt.title("Input: <s>inspecteurs<PAD><PAD><PAD>inspecteurs")
    plt.imshow(mymask)
    plt.savefig("attention_mask2.png")
#%%
from torch.nn.functional import scaled_dot_product_attention
print(tokenizer.decode(inputid))
#%%
# what happens if i specify the pad_token_id in the model, it seems like the attention_mask does not get correctly created
model.config.pad_token_id = tokenizer.pad_token_id
output = model(input_ids_with_padding, labels=labels, output_attentions=True)
# plt.imshow(output.attentions[0][0,0].detach().numpy())

#%%
output.__class__.mro()
from transformers.modeling_outputs import CausalLMOutputWithPast
output.attentions.__len__()
for attn in output.attentions:
    plt.title("Input: <s>inspecteurs<PAD><PAD>...<PAD>inspecteurs")
    plt.imshow(attn[0,0].detach().numpy()) 
    plt.show()
#%% 
model.model.__class__
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
model : LlamaModel = model 

# %%
# what does the llama model return 
# why use a llamaforcausallm model instead of just the llama model 
submodel = model.model
output = submodel(input_ids_with_padding, labels=labels, attention_mask=attention_mask, output_attentions=True)
last_hidden_state = output.last_hidden_state
last_hidden_state.shape
#%% 
output.__class__.mro()  
from transformers.modeling_outputs import BaseModelOutputWithPast
# last_hidden_state, past_key_values, hidden_states, attentions
# vs 
from transformers.modeling_outputs import CausalLMOutputWithPast
# loss, logits, past_key_values, hidden_states, attentions
# output.logits.shape 
# so the LlamaforCaussalLM model returns the logits by feeding the last hidden state into a linear layer called lm_head
# and computes the loss using the logits and the labels

# %%
model
# %%
# lets train a llamaforcausallm model on the dataset
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
    "pad_token_id": 2,             # Padding token
}
from transformers import LlamaConfig
smallconfig = LlamaConfig(**small_config)
from transformers import LlamaForCausalLM
smallmodel = LlamaForCausalLM(smallconfig)
#%%
smallmodel
# %%
# lets add some training arguments and run them

# import hfargumentparser
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
data_training_arguments = {
    "pad_to_max_length" : False,
}
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
    use_cpu=True,
)
#%% 
from transformers import DataCollatorWithPadding
from typing import Union, Optional
from transformers.tokenization_utils_base import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from dataclasses import dataclass
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
        # print(f'longest_input = {longest_input}')
        input_ids = [input_id + [self.tokenizer.pad_token_id]*(longest_input - len(input_id)) for input_id in input_ids]
        attention_masks = [attention_mask + [0]*(longest_input - len(attention_mask)) for attention_mask in attention_masks]
        input_ids_tensor = torch.tensor(input_ids)
        labels_tensor = torch.tensor(input_ids)
        return {"input_ids": input_ids_tensor, 
                "attention_mask": torch.tensor(attention_masks),
                "labels": labels_tensor}
    
data_collator = DataCollatorCustom(tokenizer)


#%%
# lets print some features to see e
for k in range(20):
    print(hg_dataset_tokenized["train"][k]["input_ids"])
    print(tokenizer.decode(hg_dataset_tokenized["train"][k]["input_ids"]))
    print('-----')

#%% 

# More efficient way to extract only needed keys
features_list = [{key: feat[key] for key in ("input_ids", "attention_mask")} 
                 for featnbr, feat in enumerate(hg_dataset_tokenized["train"]) if featnbr < 10]
collated = data_collator(features_list)
#%% 



#%% 
from transformers import Trainer
trainer = Trainer(
    model=smallmodel,
    args=training_args,
    data_collator=data_collator,
    train_dataset=hg_dataset_tokenized["train"],
    eval_dataset=hg_dataset_tokenized["validation"],
)

# %%
trainer.train()
# %%
# First, let's set the model to evaluation mode
# Generate function
def generate_text(model_to_use, prompt, max_length=50):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Generate
    outputs = model_to_use.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=3,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return the generated text
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Example usage
test_prompts = [
    "Once upon a time",
    "The future of AI",
    "In the dark forest"
]

smallmodel.eval()
for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    generations = generate_text(smallmodel, prompt)
    for i, gen in enumerate(generations, 1):
        gen = gen.replace(prompt, "") 
        print(gen)
# %%
"""
Results 
Prompt: Once upon a time
éeéesésésationsantéantéationéeéeableéesantsantesitéitendantitééeéeséeséesiquesantéesésiresiluéesreéséeisableésitééesééant
antéeinésaitéatéerantéeitationéuléesantéend- déitééser-anté éageég-éeatésésantééeéeés réationéesenté
isementéeséeéséireésauxésisementantendéeé réeresantantilentéeéisant-esaitéreationés-éonesuéiteé éenté

Prompt: The future of AI
-ésantantéonentéueursinéeséséeéeeantéeéantitésentieantéesésésééeseséeéantéséeésementéeséitééeséeséesée
-séentitéséèermentéeiss-uéeséeendentéséationursnationsitéséeséesulageéementatéésonégsésireée-ulué
éurursesséeantesantéeséueésésé-esenderéséesentiéoneséeé réuritéatéeséesentésesableséulééséeit

Prompt: In the dark forest
iéesaitéesendéeséséentéueéi-ationsitéesationsireéséeantéséeantitéeréesentanturéatésantésuésantésésésantésable
éignentquatementéséséseréisentueéesageéeséeantéonuléséséeanteanté réentéueésée réementéeés rééeéeéagei
ieitéitééesééeéulantentééré prééiteéisséationsé réantonesentiteéeéantsésentisantééitéiséeséséeerées
"""
# lets add the end of sentence token to the tokenizer

def tokenize_v2(examples, max_length=512):
    # Add EOS token by setting add_special_tokens=True
    outputs = tokenizer(
        examples['text'],
        padding="longest",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True  # This ensures special tokens like EOS are added
    )
    
    # If you specifically want to force EOS at the end of each sequence:
    if outputs['input_ids'][-1] != tokenizer.eos_token_id:
        # Only add if there's room and it's not already there
        if len(outputs['input_ids']) < max_length:
            outputs['input_ids'].append(tokenizer.eos_token_id)
            outputs['attention_mask'].append(1)
        else:
            # Truncate and replace the last token with EOS
            outputs['input_ids'][-1] = tokenizer.eos_token_id
    return outputs

hg_dataset_tokenized_v2 = hg_dataset.map(tokenize_v2)
# %%

# %%
smallmodel.init_weights()
trainer_v2 =  Trainer(
    model=smallmodel,
    args=training_args,
    data_collator=data_collator,
    train_dataset=hg_dataset_tokenized["train"],
    eval_dataset=hg_dataset_tokenized["validation"],
)
trainer_v2.train()

# %%
smallmodel.eval()
for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    generations = generate_text(smallmodel, prompt)
    break 
# %%
input =tokenizer("hello there !", return_tensors="pt")
# Pass the input_ids tensor specifically
new_str = ""
output = smallmodel(input['input_ids'])
# If your model expects attention masks too:
output = smallmodel(input_ids=input['input_ids'], attention_mask=input['attention_mask'])

output_logits = output.logits
best_token = torch.argmax(output_logits[0, -1, :]).item()
print(tokenizer.decode(best_token))



# If you want to add some randomness, here's a version with temperature:
#%%
max_length = 50
temperature = 0.7
generated_tokens = ""
input_ids = tokenizer(prompt, return_tensors="pt")['input_ids']
generated_tokens = input_ids[0].tolist()

for _ in range(max_length):
    outputs = smallmodel(input_ids=torch.tensor([generated_tokens], dtype=torch.long))
    next_token_logits = outputs.logits[:, -1, :] / temperature
    
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
    
    # Sample from the distribution
    next_token = torch.multinomial(probs[0], num_samples=1).item()
    
    generated_tokens.append(next_token)
    
    if next_token == tokenizer.eos_token_id:
        print("EOS token found, stopping generation")
        break
generated_tokens = [int(token) for i, token in enumerate(generated_tokens) if i >= len(input_ids)]
tokenizer.decode(generated_tokens)  
#%%%

