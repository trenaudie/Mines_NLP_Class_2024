#%% 
from pathlib import Path
import json 
from tqdm import tqdm
from bs4 import BeautifulSoup
from itertools import islice
from icecream import ic
import re
from project3.get_response import get_response
DATA_DIR = Path("project3/data")
FILTERING_PROMPT = open("project3/prompts/preprocessing_order.txt", "r").read()
CLASSIFICATION_PROMPT = open("project3/prompts/classification_order.txt", "r").read()

def extract_json(response:str):
    json_matcher = r'(\`\`\`json\n[\s\S]*?\n\`\`\`)'
    json_matched = re.findall(json_matcher, response)
    if len(json_matched) == 0:
        raise ValueError("No JSON found in the response")
    cleaned = json_matched[0].replace('```json', '').replace('```', '')
    cleaned = cleaned.strip()
    return json.loads(cleaned)


def clean_and_parse_json(unclean_string:str):
    import re
    json_matcher = r'(\`\`\`json\n[\s\S]*?\n\`\`\`)'
    json_matched = re.findall(json_matcher, unclean_string)
    if len(json_matched) == 0:
        raise ValueError("No JSON found in the response")
    cleaned_string = json_matched[0].strip().strip("'").replace('\\n', '\n').replace('\\"', '"')

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
for file in tqdm(islice(DATA_DIR.iterdir(), 30)):
    save_path = RESULTS_DIR / f"{file.name}_result.json"
    if save_path.exists():
        print(f'Already processed {file.name}')
        continue
    else:
        print(f'Processing {file.name}')
    with open(file, "r") as f:
        text = f.read()
    soup = BeautifulSoup(text, "html.parser")
    textgotten = soup.get_text()
    full_filtered_prompt = f"{FILTERING_PROMPT}\n\n{textgotten}"
    print(f'extracting text using llm')
    try:
        response = get_response(full_filtered_prompt)
    except Exception as e:
        print(response)
        print()
        continue
    obj = extract_json(response)
    print(f'classifying text using llm')
    full_classification_prompt = f"{CLASSIFICATION_PROMPT}\n\n{obj}"
    try:
        response = get_response(full_classification_prompt)
    except Exception as e:
        print(response)
        print()
        continue
    clean_response = clean_and_parse_json(response)
    all_responses[file.name] = clean_response
    with open(save_path, "w") as f:
        json.dump(clean_response, f)
    ic(clean_response)
# %%

# script to read from results and concatenate into list of dicts 
results_dir = Path("project3/results")
all_results = []
for file in results_dir.iterdir():
    with open(file, "r") as f:
        all_results.append(json.load(f))
# script to save all_results as jsonl file 
with open("project3/all_results.jsonl", "w") as f:
    for result in all_results:
        f.write(json.dumps(result) + "\n")
# %%


# %%
