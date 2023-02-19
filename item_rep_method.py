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

