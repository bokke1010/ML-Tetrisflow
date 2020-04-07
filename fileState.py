import json
from ast import literal_eval

def save_state(state, filename):
    stringState = {}
    for key in state:
        stringState[str(key)] = state[key]
    with open(filename, 'w') as file:
        json.dump(stringState, file)

def load_state(filename):
    data = None
    with open(filename, 'r') as file:
        data = json.load(file)
    evaluatedData = {}
    for key in data:
        evaluatedData[literal_eval(key)] = data[key]
    return evaluatedData
