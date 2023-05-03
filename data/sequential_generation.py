import time
from tqdm import tqdm
import json
import random

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

# generate time-sensitive sequential data
with open("remapped_sequential_data.txt", "w") as f:
    for sequences in remapped_sequential_data:
        f.write(sequences)


#####################
# generate other sequential setting data
with open("remapped_sequential_data.txt", "r") as f:
    data = f.read()


data = data.split("\n")[:-1]
randomize_data = data.copy()
random.shuffle(randomize_data)
data = {d.split(" ")[0]: d.split(" ")[1:] for d in data}


##### different sorting

# sort short to long
short_to_long_data = sorted(data.items(), key=lambda x: len(x[1]))
short_to_long_data = {k: v for k, v in short_to_long_data}


# sort long to short
long_to_short_data = sorted(data.items(), key=lambda x: len(x[1]), reverse=True)
long_to_short_data = {k: v for k, v in long_to_short_data}

# randomize
randomize_data = {d.split(" ")[0]: d.split(" ")[1:] for d in randomize_data}


##### generate file based on the ordering
# generate data for short to long
short_to_long_map = {}
new_id = 0
for k, v in short_to_long_data.items():
    for item in v[:-2]:
        if item not in short_to_long_map:
            new_id += 1
            short_to_long_map[item] = new_id

remapped_short_to_long_data = {}
for k, v in short_to_long_data.items():
    remapped_items = []
    for item in v:
        if item in short_to_long_map:
            remapped_items.append(short_to_long_map[item])
        else:
            remapped_items.append(item)
    remapped_short_to_long_data[k] = remapped_items


short_to_long_remapped_sequential_data = []
for k, v in remapped_short_to_long_data.items():
    one_line = str(k) + " " + " ".join([str(a) for a in v]) + "\n"
    short_to_long_remapped_sequential_data.append(one_line)


with open("short_to_long_remapped_sequential_data.txt", "w") as f:
    for sequences in short_to_long_remapped_sequential_data:
        f.write(sequences)


# generate data for long to short
long_to_short_map = {}
new_id = 0
for k, v in long_to_short_data.items():
    for item in v[:-2]:
        if item not in long_to_short_map:
            new_id += 1
            long_to_short_map[item] = new_id

remapped_long_to_short_data = {}
for k, v in long_to_short_data.items():
    remapped_items = []
    for item in v:
        if item in long_to_short_map:
            remapped_items.append(long_to_short_map[item])
        else:
            remapped_items.append(item)
    remapped_long_to_short_data[k] = remapped_items

long_to_short_remapped_sequential_data = []
for k, v in remapped_long_to_short_data.items():
    one_line = str(k) + " " + " ".join([str(a) for a in v]) + "\n"
    long_to_short_remapped_sequential_data.append(one_line)

with open("long_to_short_remapped_sequential_data.txt", "w") as f:
    for sequences in long_to_short_remapped_sequential_data:
        f.write(sequences)


# generate data for randomize
randomize_map = {}
new_id = 0
for k, v in randomize_data.items():
    for item in v[:-2]:
        if item not in randomize_map:
            new_id += 1
            randomize_map[item] = new_id

remapped_randomize_data = {}
for k, v in randomize_data.items():
    remapped_items = []
    for item in v:
        if item in randomize_map:
            remapped_items.append(randomize_map[item])
        else:
            remapped_items.append(item)
    remapped_randomize_data[k] = remapped_items


randomize_remapped_sequential_data = []
for k, v in remapped_randomize_data.items():
    one_line = str(k) + " " + " ".join([str(a) for a in v]) + "\n"
    randomize_remapped_sequential_data.append(one_line)


with open("randomize_remapped_sequential_data.txt", "w") as f:
    for sequences in randomize_remapped_sequential_data:
        f.write(sequences)

