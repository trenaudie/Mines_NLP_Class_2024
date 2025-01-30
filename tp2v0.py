# %% [markdown]

# %% [markdown]
# ## Données sources
# 
# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


# %%
words = open('data/codes_mots/civil_mots.txt', 'r').read().splitlines()
nb_words = len(words)
print("nb_words =", nb_words)

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
# ## Construction du jeu de données pour l'entraînement

# %%
def build_dataset(words:list, context_size:int):
    """Build the dataset of the neural net for training.

    Parameters:
        words: list of words of our data corpus
        context_size: how many characters we take to predict the next one

    Returns:
        X: inputs to the neural net
        Y: labels
    """
    X, Y = [], []
    for w in words:
        #print(w)
        context = [0] * context_size
        for ch in w + '.':
            ix = ctoi[ch]
            X.append(context)
            Y.append(ix)
            #print(''.join(itoc[i] for i in context), '--->', itoc[ix])
            context = context[1:] + [ix] # crop and append
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    #print(X.shape, Y.shape)
    return X, Y

# %% [markdown]
# Les mots du code civil générent un jeu d'entraînement avec les entrées `X` de dimension 2 de forme (67652, 3), soit 67652 contextes de 3 caractères différents et pour les labels `Y` 67652 caractères suivants.

# %%
context_size = 3
X, Y = build_dataset(words, context_size)
print("X.shape =", X.shape)
print("Y.shape =", Y.shape)
print(X[:5])
print(Y[:5])



# %% [markdown]
# ## Réseau complet et entraînement

# %% [markdown]
# ### Architecture

# %%
e_dims = 10  # Dimensions des embeddings
INT_SIZE = 200
print("nb_chars =", nb_chars)
print("e_dims =", e_dims)
g = torch.Generator().manual_seed(2147483647) # for reproducibility


# %% [markdown]
# ### Jeux d'entraînement, de développement et de test

# %%
# 80%, 10%, 10%
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

context_size = 3
Xtr, Ytr = build_dataset(words[:n1], context_size=context_size)
Xdev, Ydev = build_dataset(words[n1:n2], context_size=context_size)
Xte, Yte = build_dataset(words[n2:], context_size=context_size)



# %% [markdown]
# ### Entraînement
# start a new wandb run to track this script
LR = 0.05
import wandb
wandb.login()

#%%
wandb.init(
    # set the wandb project where this run will be logged
    project="nlp_project_lesmines",
    # track hyperparameters and run metadata
    config={
    "learning_rate":LR,
    "architecture": "MLP_l200l",
    "dataset": "civil_mots",
    "epochs": None,
    "iterations": 100_000,
    "batch_size": 32,
    }
)


# %%
lossi = []
stepi = []

# %%

class MLPv0_with_Embedding(torch.nn.Module):
    def __init__(self, nb_chars, e_dims, context_size, INT_SIZE):
        super().__init__()
        self.embedding = torch.nn.Embedding(nb_chars, e_dims)
        self.fc1 = torch.nn.Linear(context_size*e_dims, INT_SIZE)
        self.fc2 = torch.nn.Linear(INT_SIZE, nb_chars)
    def forward(self, x):
        emb = self.embedding(x)
        emb_reshaped = emb.view(-1, context_size*e_dims)
        h = F.relu(self.fc1(emb_reshaped))
        logits = self.fc2(h)
        return logits
    
import torch
import torch.nn.functional as F

class MLPv1_with_Embedding(torch.nn.Module):
    def __init__(self, 
                 nb_chars, 
                 e_dims, 
                 context_size, 
                 hidden_sizes=[512, 1024, 512, 256],
                 dropout_rate=0.2):
        super().__init__()
        
        # Embedding layer
        self.embedding = torch.nn.Embedding(nb_chars, e_dims)
        
        # Calculate input size for first linear layer
        input_size = context_size * e_dims
        
        # Create list to hold all layers
        layers = []
        
        # Input layer
        layers.extend([
            torch.nn.Linear(input_size, hidden_sizes[0]),
            torch.nn.LayerNorm(hidden_sizes[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate)
        ])
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.extend([
                torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                torch.nn.LayerNorm(hidden_sizes[i+1]),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate)
            ])
            
        # Output layer
        layers.append(torch.nn.Linear(hidden_sizes[-1], nb_chars))
        
        # Create sequential model
        self.network = torch.nn.Sequential(*layers)
        
        # Initialize weights using Kaiming initialization
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.LayerNorm):
                if module.elementwise_affine:
                    torch.nn.init.ones_(module.weight)
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Get embeddings
        emb = self.embedding(x)
        
        # Reshape embeddings
        batch_size = x.size(0)
        emb_reshaped = emb.view(batch_size, -1)
        
        # Pass through network
        logits = self.network(emb_reshaped)
        
        return logits

# Example usage:
# %% 

model = MLPv1_with_Embedding(nb_chars, e_dims, context_size)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)



#%% 
def train_model(model, optimizer, Xtr, Ytr, nb_chars, nb_iterations=50_000):
    for i in tqdm(range(nb_iterations)):
        # minibatch construct
        ix = torch.randint(0, Xtr.shape[0], (32,))
        minibatch = Xtr[ix] #shape (32, 3)
        # forward pass
        logits = model(minibatch).view(-1, nb_chars) # (32, nb_chars)
        loss = F.cross_entropy(logits, Ytr[ix])
        accuracy = (logits.argmax(dim=1) == Ytr[ix]).float().mean()
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stepi.append(i)
        lossi.append(loss.log10().item())
        wandb.log({"loss": loss.item(), "step": i})
        wandb.log({"accuracy": accuracy.item(), "step": i})


# %%
train_model(model, optimizer, Xtr, Ytr, nb_chars, nb_iterations=100_000)
plt.plot(stepi, lossi)



logits = model(Xtr).view(-1, nb_chars) # (32, nb_chars)
loss = F.cross_entropy(logits, Ytr)
loss

# %%
logits = model(Xdev).view(-1, nb_chars) # (32, nb_chars)    
loss = F.cross_entropy(logits, Ydev)
loss

# %%
# visualize dimensions 0 and 1 of the embedding matrix C for all characters
C = model.embedding.weight
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itoc[i], ha="center", va="center", color='white')
plt.grid('minor')

# %% [markdown]
# ## Utilisation du modèle: génération de mots

# %%
context = [0] * context_size
C = model.embedding.weight
W1 = model.fc1.weight
b1 = model.fc1.bias
W2 = model.fc2.weight
b2 = model.fc2.bias
C[torch.tensor([context])].shape

# %%
# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * context_size # initialize with all ...
    while True:
        logits = model(torch.tensor(context).view(1,-1))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itoc[i] for i in out))

# %% [markdown]
# ```
# aves.
# défester.
# pièces.
# sociale.
# apprétaire.
# succeplément.
# mes.
# trouvoire.
# assuu.
# tuteurressorts.
# momplies.
# justitualité.
# exprive.
# herces.
# délégales.
# memblateurt.
# résent.
# qu'aucunt.
# pe.
# norement.
# ```

# %%

# tokenizer 

from transformers import AutoTokenizer 
from huggingface_hub import notebook_login, login
login(token = "enter secret here")

# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer.get_vocab().__len__()
vocab_size = tokenizer.get_vocab().__len__()
e_dims = 64 
context_size = 3
model = MLPv1_with_Embedding(vocab_size, e_dims, context_size, [24,24,24,24])

# %% 

# build dataset 

from datasets import Dataset
from pathlib import Path
import pandas as pd
from datasets import load_dataset
codes_mots_dir = 'data/codes_mots/'
words = []
for file in Path(codes_mots_dir).rglob('*.txt'):
    with open(file, 'r') as f:
        words.extend(f.read().splitlines())
nb_words = len(words)
print("nb_words =", nb_words)
# words is a list of words
dataframe = pd.DataFrame(words, columns=["text"])
dataframe.to_json(f"data_preprocessed/codes_mots_preprocessed.jsonl", orient= "records", lines=True)
words_dataset = load_dataset("json", data_files="data_preprocessed/codes_mots_preprocessed.jsonl", split="train")
words_dataset
#%%
def my_map_function(example:dict):
    return tokenizer(example["text"], padding=None, truncation=True)
dataset2 = words_dataset.map(my_map_function, batched=False)
# %%

nb_iterations = 2
for i in tqdm(range(nb_iterations)):
    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (32,))
    minibatch = Xtr[ix] #shape (32, 3)
    # forward pass
    logits = model(minibatch).view(-1, nb_chars) # (32, nb_chars)
    loss = F.cross_entropy(logits, Ytr[ix])
    accuracy = (logits.argmax(dim=1) == Ytr[ix]).float().mean()
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    stepi.append(i)
    lossi.append(loss.log10().item())
    wandb.log({"loss": loss.item(), "step": i})
    wandb.log({"accuracy": accuracy.item(), "step": i})

# %%
