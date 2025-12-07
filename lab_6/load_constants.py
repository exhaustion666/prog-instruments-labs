import json

def load_constants(filename: str) -> dict:
    """
    Loads constants from a JSON file.

    args:
        filename (str): The name of the JSON file containing constants.

    return:
        dict: A dictionary containing the loaded constants.
    """
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)
