import os 
from dotenv import load_dotenv
load_dotenv()
def env_is_debug():
    return os.getenv("DEBUG") == "True" 
