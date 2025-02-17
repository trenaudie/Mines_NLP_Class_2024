#%%%
import pandas as pd 
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch
from tqdm import tqdm
from itertools import islice
import wandb
import random

LR = 0.02
ARCHI = "MLP_l10l"
DATASET = "codes"
EPOCHS = 1

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="nlp_project_lesmines",

    # track hyperparameters and run metadata
    config={
    "learning_rate": LR,
    "architecture": ARCHI,
    "dataset": DATASET,
    "epochs": EPOCHS,
    }
)



codes_dir = Path("codes")
words_list = []
for md_file in codes_dir.glob("*.md"):
    file_content = md_file.read_text()
    words_list.extend(file_content.split())

words_set = set(words_list)
vocabulary = list(words_set)
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
V = len(vocabulary)

model = nn.Sequential(
    nn.Linear(V, 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, V),
)

def test_model():
    sentence = "réalisation du projet"
    encoded = torch.zeros(3,V)
    for t, word in enumerate(sentence.split()):
        encoded[t, words_list.index(word)] = 1
    output = model(encoded)
    target = torch.tensor([words_list.index(word) for word in sentence.split()])
    lossfn= nn.CrossEntropyLoss()
    loss = lossfn(output, target)
    return loss


# Verification que la fonction CrossEntropyLoss de PyTorch fait bien une somme des logprobabilités. 
def mycrossentropy(output, target):
    output_softmax = nn.functional.softmax(output, dim=1)
    assert output_softmax.shape[0] == len(target)
    return -torch.log(output_softmax[torch.arange(len(target)),target]).sum() / len(target)

def transform_to_prediction(output):  
    prediction = torch.argmax(output, dim=1)
    prediction_str = " ".join([vocabulary[idx] for idx in prediction])
    return prediction_str
# %%
def test_model(model, words_list_test, lossfn):
    test_loss = []
    for i in tqdm(range(len(words_list_test) - 1)):
        word_i = words_list_test[i]
        word_j = words_list_test[i + 1]
        encoded = torch.zeros(1, V)
        encoded[0, word_to_idx[word_i]] = 1
        output = model(encoded)
        target = torch.tensor([word_to_idx[word_j]])
        loss = lossfn(output, target)
        test_loss.append(loss.item())
    return test_loss
#%%

split_percentage = 0.8 
words_list_train = words_list[:int(len(words_list)*split_percentage)]
words_list_test = words_list[int(len(words_list)*split_percentage):]
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
training_loss = [] 
for i in tqdm(islice(range(len(words_list_train)-1),2000)):
    word_i = words_list_train[i]
    word_j = words_list_train[i+1]
    encoded = torch.zeros(1,V)
    encoded[0, word_to_idx[word_i]] = 1
    output = model(encoded)
    target = torch.tensor([word_to_idx[word_j]])
    loss = lossfn(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    training_loss.append(loss.item())
    wandb.log({"train_loss": loss.item()})
    if i % 100 == 0:
        print(f"{i} - loss: {loss.item()}")
    if i%200 == 0:
        test_loss = test_model(model, words_list_test)
        wandb.log({"test_loss": sum(test_loss)/len(test_loss)})
    
wandb.finish()


# run test 





# %% 
# bigram model is simply the conditional probability model 
"""
P(w_i | w_j) = sum( w_i -> w_j ) / sum(w_i)
Compute all the probabilities during training 

Inference: 
w_i -> argmax_a P(a | w_i )
"""

#train 
word_counts = np.zeros((V,V))  
for i in islice(tqdm(range(len(words_list_train)-1)),None):
    word_i = words_list_train[i]
    word_j = words_list_train[i+1]
    word_to_idx_i = word_to_idx[word_i]
    word_to_idx_j = word_to_idx[word_j]
    word_counts[word_to_idx_i, word_to_idx_j] += 1
#%%
# inference
for i in range(10):
    word_i = words_list_test[i]
    word_j = words_list_test[i+1]
    word_i_idx = word_to_idx[word_i]
    word_j_idx = word_to_idx[word_j]
    preds = word_counts[word_i_idx]
    top_pred_idx = np.argmax(preds)
    loss = - np.log(preds[word_j_idx])
    print(loss)



    
    


# %%
