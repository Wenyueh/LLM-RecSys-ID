import random
import os
import json
import time
import math


def random_number(max_number):
    randomized_number = list(range(max_number))
    random.shuffle(randomized_number)
    mapping = {str(i): str(v) for i, v in enumerate(randomized_number)}
    return lambda x: mapping[x]


def random_one_token(tokenizer):
    token_range = list(range(20, 32000))
    random.shuffle(token_range)
    mapping = {str(i): "<pad>" + tokenizer.decode(v) for i, v in enumerate(token_range)}
    return lambda x: mapping[x]


def random_two_token(tokenizer):
    one_token_range = list(range(20, 32000))
    two_token_range = list(range(20, 32000))
    random.shuffle(one_token_range)
    random.shuffle(two_token_range)
    mapping = {
        str(i): tokenizer.decode([v1, v2])
        for i, (v1, v2) in enumerate(zip(one_token_range, two_token_range))
    }
    return lambda x: mapping[x]


def no_tokenization():
    # has to be the remodeled tokenizer
    mapping = {
        str(i): "<extra_id_{}>".format(str(v)) for i, v in enumerate(range(30000))
    }
    return lambda x: mapping[x]


def composition_resolution(item_index, resolution, overlap):
    assert type(item_index) == str
    assert resolution > 0
    assert resolution > overlap
    sub_indices = []
    number_of_sub_indices = math.ceil(len(item_index) / (resolution - overlap))
    for i in range(number_of_sub_indices):
        sub_indices.append(
            item_index[
                i * (resolution - overlap) : (i * (resolution - overlap) + resolution)
            ]
        )
    split_zero_sub_indices = []
    for i in range(len(sub_indices)):
        if sub_indices[i].startswith("0"):
            zero_index = sub_indices[i]
            j = 0
            while zero_index[j:].startswith("0"):
                split_zero_sub_indices.append("0")
                j += 1
            if zero_index[j:] != "":
                split_zero_sub_indices.append(zero_index[j:])
        else:
            split_zero_sub_indices.append(sub_indices[i])

    return split_zero_sub_indices


def unit_test_composition_resolution():
    ######## no overlap ########
    ######## no overlap ########
    # test 1
    item_index = "1230"
    resolution = 2
    overlap = 0
    sub_indices = composition_resolution(item_index, resolution, overlap)
    assert sub_indices == ["12", "30"]

    # test 2
    item_index = "12301"
    resolution = 2
    overlap = 0
    sub_indices = composition_resolution(item_index, resolution, overlap)
    assert sub_indices == ["12", "30", "1"]

    # test 3
    item_index = "12301"
    resolution = 3
    overlap = 0
    sub_indices = composition_resolution(item_index, resolution, overlap)
    assert sub_indices == ["123", "0", "1"]

    # test 4
    item_index = "123010"
    resolution = 3
    overlap = 0
    sub_indices = composition_resolution(item_index, resolution, overlap)
    assert sub_indices == ["123", "0", "10"]

    ######## with overlap = 1 ########
    ######## with overlap = 1 ########
    # test 5
    item_index = "123010"
    resolution = 2
    overlap = 1
    sub_indices = composition_resolution(item_index, resolution, overlap)
    assert sub_indices == ["12", "23", "30", "0", "1", "10", "0"]

    # test 6
    item_index = "123010"
    resolution = 3
    overlap = 1
    sub_indices = composition_resolution(item_index, resolution, overlap)
    assert sub_indices == ["123", "301", "10"]

    # test 7
    item_index = "123010"
    resolution = 3
    overlap = 1
    sub_indices = composition_resolution(item_index, resolution, overlap)
    assert sub_indices == ["1230", "0", "10"]

    # test 8
    item_index = "123010"
    resolution = 5
    overlap = 1
    sub_indices = composition_resolution(item_index, resolution, overlap)
    assert sub_indices == ["12301", "10"]

    ######## with overlap = 2 ########
    ######## with overlap = 2 ########
    # test 8
    item_index = "1"
    resolution = 4
    overlap = 2
    sub_indices = composition_resolution(item_index, resolution, overlap)
    assert sub_indices == ["1"]

    # test 9
    item_index = "12"
    resolution = 4
    overlap = 2
    sub_indices = composition_resolution(item_index, resolution, overlap)
    assert sub_indices == ["12"]


if __name__ == "__main__":
    item_index = "12"
    resolution = 4
    overlap = 2
    sub_indices = composition_resolution(item_index, resolution, overlap)
    print(sub_indices)
