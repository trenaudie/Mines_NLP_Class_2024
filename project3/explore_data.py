#%% 
from pathlib import Path
import json 
from tqdm import tqdm
from bs4 import BeautifulSoup
from itertools import islice
from project3.get_response import get_response
DATA_DIR = Path("project3/data")

FILTERING_PROMPT = open("project3/prompts/preprocessing_order.txt", "r").read()
CLASSIFICATION_PROMPT = open("project3/prompts/classification_order.txt", "r").read()

def extract_json(response:str):
    cleaned = response.replace('```json', '').replace('```', '')
    cleaned = cleaned.strip()
    return json.loads(cleaned)


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

# %%
RESULTS_DIR = Path("project3/results")
RESULTS_DIR.mkdir(exist_ok=True)
all_responses = dict()
for file in tqdm(islice(DATA_DIR.iterdir(), 1,2)):
    print(f'Processing {file.name}')
    with open(file, "r") as f:
        text = f.read()
    soup = BeautifulSoup(text, "html.parser")
    textgotten = soup.get_text()
    full_filtered_prompt = f"{FILTERING_PROMPT}\n\n{textgotten}"
    response = get_response(full_filtered_prompt)
    obj = extract_json(response)
    full_classification_prompt = f"{CLASSIFICATION_PROMPT}\n\n{obj}"
    response = get_response(full_classification_prompt)
    print(response)
    clean_response = clean_and_parse_json(response)
    all_responses[file.name] = clean_response
    with open(RESULTS_DIR / f"{file.name}_result.json", "w") as f:
        json.dump(clean_response, f)
    break 
# %%
