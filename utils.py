import json


def read_json(json_file: str):
    with open(json_file, "r") as f:
        return json.load(f)


def save_json(json_file: str, data: dict):
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
