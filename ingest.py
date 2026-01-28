import json

docs = []
with open("data/finance_guidelines.jsonl", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            docs.append(json.loads(line))