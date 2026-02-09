import json
import os
from dotenv import load_dotenv

load_dotenv()

# add data path environment variable
DATA_PATH = os.getenv("DATA_PATH")
docs = []
with open(DATA_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            docs.append(json.loads(line))