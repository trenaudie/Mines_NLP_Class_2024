#%% 
from pathlib import Path 
from dotenv import load_dotenv
load_dotenv()
import requests
import json
import os
openrouter_api_key = os.getenv("OPEN_ROUTER_KEY")
model_name = "google/gemini-2.0-flash-exp:free" # hit the limit
model_name = "google/gemini-2.0-pro-exp-02-05:free" # very slow
model_name = "meta-llama/llama-3.3-70b-instruct:free" # very reliable!
print(openrouter_api_key )

def get_response(prompt:str) -> str:
  try:
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
      }),
      timeout=15  # Add 15 second timeout
    )
  except requests.exceptions.Timeout:
    raise TimeoutError("Request timed out after 15 seconds")
    
  try:
    content = response.json()['choices'][0]['message']['content']
  except Exception as e:
    print(response.json())
    raise e
  return content
