import time
from tqdm import tqdm
import json


with open("sequential_data.txt", "r") as f:
    data = f.read()

data = data.split("\n")[:-1]
data = {d.split(" ")[0]: d.split(" ")[1:] for d in data}


all_train_sequence = [a for k, v in data.items() for a in v[:-2]]
all_sequence = [a for k, v in data.items() for a in v]


remap = {}
index = 1
for item in all_train_sequence:
    if item not in remap:
        remap[item] = str(index)
        index += 1
for item in all_sequence:
    if item not in remap:
        remap[item] = str(index)
        index += 1


remapped_data = {}
for k, v in data.items():
    remapped_sequence = []
    remapped_data[k] = [remap[a] for a in v]


remapped_sequential_data = []
for k, v in remapped_data.items():
    one_line = k + " " + " ".join(v) + "\n"
    remapped_sequential_data.append(one_line)

with open("remapped_sequential_data.txt", "w") as f:
    for sequences in remapped_sequential_data:
        f.write(sequences)
