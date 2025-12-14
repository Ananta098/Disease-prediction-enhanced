import csv
import json
import ast
from collections import defaultdict

data = defaultdict(list)

with open("processed_disease_data.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        symptoms = ast.literal_eval(row["symptoms"])
        data[row["disease"]].append(symptoms)

with open("diseases.json", "w") as f:
    json.dump(data, f, indent=2)

print("JSON file created: diseases.json")
