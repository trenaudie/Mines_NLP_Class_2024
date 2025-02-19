#%% 
from pathlib import Path
import json 
from tqdm import tqdm
DATA_DIR = Path("project3/data")
file = DATA_DIR.iterdir().__next__()
str(file)
# %%
len(list(DATA_DIR.iterdir()))
# %%
from project3.get_response import get_response

file = "1994-08-03_AP-auto_refonte_pixtral.html"
file = DATA_DIR / file
file_text= open(file, "r").read()
# %%
file_text
from bs4 import BeautifulSoup
soup = BeautifulSoup(file_text, "html.parser")
# %%
soup_text = soup.get_text()
# %%
soup_text
# %%

# %%
from pathlib import Path

# select first file
file = list(DATA_DIR.iterdir())[0]

# read file
with open(file, "r") as f:
    text = f.read()
print(text)
soup = BeautifulSoup(text, "html.parser")
# %%
textgotten = soup.get_text()
# %%
print(textgotten)
filtering_prompt = open("project3/prompts/preprocessing_order.txt", "r").read()
full_prompt = f"{filtering_prompt}\n\n{textgotten}"

# %%
response = get_response(full_prompt)
# %%
import json 
def extract_json(response:str):
    cleaned = response.replace('```json', '').replace('```', '')
    cleaned = cleaned.strip()
    return json.loads(cleaned)
# %%


def clean_and_parse_json(unclean_string:str):
    import re
    cleaned_string = unclean_string.strip().strip("'").replace('\\n', '\n').replace('\\"', '"')

    cleaned_string = cleaned_string.replace('```json', '').replace('```', '').strip()

    cleaned_string = cleaned_string.replace('\\n', '\n').replace('\\"', '"')    

    cleaned_string = re.sub(r',\s*}', '}', cleaned_string)
    try:
        python_object = json.loads(cleaned_string)
        return python_object
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

clean_string = clean_and_parse_json(response)
# %%
RESULTS_DIR = Path("project3/results")
RESULTS_DIR.mkdir(exist_ok=True)
all_responses = dict()
for file in DATA_DIR.iterdir():
    with open(file, "r") as f:
        text = f.read()
    soup = BeautifulSoup(text, "html.parser")
    textgotten = soup.get_text()
    full_filtered_prompt = f"{filtering_prompt}\n\n{textgotten}"
    response = get_response(full_filtered_prompt)
    obj = extract_json(response)
    full_classification_prompt = f"{classification_prompt}\n\n{obj}"
    response = get_response(full_classification_prompt)
    print(response)
    clean_response = clean_and_parse_json(response)
    all_responses[file.name] = clean_response
    with open(RESULTS_DIR / f"{file.name}_result.json", "w") as f:
        json.dump(clean_response, f)
