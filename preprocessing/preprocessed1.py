#%%
import nltk
from tqdm import tqdm
from datetime import datetime
import spacy
import string
from pathlib import Path
codes_dir = Path("codes")
assert codes_dir.exists()
words_list = []
files = codes_dir.glob("*.md")


# %%
PUNCT_TO_REMOVE = string.punctuation
PUNCT_TO_REMOVE = PUNCT_TO_REMOVE.replace("'", "").replace(".","") # don't remove '
PUNCT_TO_REMOVE
# %%
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

words_final = ""
for file in tqdm(files):
    with open(file, "r") as f:
        words = f.read() 
        words_no_punct = remove_punctuation(words)
        words_final += words_no_punct
code_preprocessed_dir = Path("code_preprocessed")
code_preprocessed_dir.mkdir(exist_ok=True)
today_str  = datetime.today().strftime('%m-%d')
with open(code_preprocessed_dir / f"code_preprocessed_{today_str}.txt", "w") as f:
    f.write(words_final)

# %%
