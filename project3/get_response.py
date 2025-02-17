#%% 
from pathlib import Path 
from dotenv import load_dotenv
load_dotenv()
import requests
import json
import os
openrouter_api_key = os.getenv("OPEN_ROUTER_KEY")
model_name = "google/gemini-2.0-flash-exp:free"
print(openrouter_api_key )

def get_response(prompt:str) -> str:
  response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
      "Authorization": f'Bearer {openrouter_api_key}',
    },
    data=json.dumps({
      "model": model_name,
      "messages": [
        {
          "role": "user",
          "content":prompt
        }
      ]
    })
  )
  content = response.json()['choices'][0]['message']['content']
  return content
