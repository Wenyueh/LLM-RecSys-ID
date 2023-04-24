from sklearn.cluster import SpectralClustering
import random
import os
import json
import time
import math
import numpy as np
import torch
import gzip
from tqdm import tqdm
from CF_index import (
    construct_indices_from_cluster,
    within_category_spectral_clustering,
    construct_indices_from_cluster_optimal_width,
)
import argparse
from collections import defaultdict, Counter
from itertools import combinations
import warnings
import pickle as pkl


warnings.filterwarnings("ignore")


def random_number(max_number, min_number=0):
    randomized_number = list(range(min_number, max_number))
    random.shuffle(randomized_number)
    mapping = {str(i): str(v) for i, v in enumerate(randomized_number)}
    return lambda x: mapping[x]


def random_number_remove_zero(max_number, min_number=0):
    randomized_number = [
        a for a in list(range(min_number, max_number)) if "0" not in str(a)
    ]
    random.shuffle(randomized_number)
    mapping = {str(i): str(v) for i, v in enumerate(randomized_number)}
    return lambda x: mapping[x]


def no_tokenization(args):
    if args.data_order != "remapped_sequential":
        # has to be the remodeled tokenizer
        mapping = {
            str(i): "<extra_id_{}>".format(str(v)) for i, v in enumerate(range(30000))
        }
    else:
        mapping = {
            str(i): str(i) + "<extra_id_{}>".format(str(v))
            for i, v in enumerate(range(30000))
        }
    return lambda x: mapping[x]


def amazon_asin(args):
    id2item_dir = args.data_dir + args.task + "/datamaps.json"
    with open(id2item_dir, "r") as f:
        datamaps = json.load(f)
    id2item = datamaps["id2item"]

    return id2item


def load_meta(args):
    id2item_dir = args.data_dir + args.task + "/datamaps.json"

    def parse(path):
        g = gzip.open(path, "r")
        for l in g:
            yield eval(l)

    if args.task != "yelp":
        meta_dir = args.data_dir + args.task + "/meta.json.gz"

        meta_data = []
        for meta in parse(meta_dir):
            meta_data.append(meta)

        meta_dict = {}
        for i, meta_item in enumerate(meta_data):
            meta_dict[meta_item["asin"]] = i
    else:
        meta_dir = args.data_dir + args.task + "/meta_data.pkl"
        with open(meta_dir, "rb") as f:
            meta_data = pkl.load(f)
        meta_dict = {}
        for i, meta_item in enumerate(meta_data):
            meta_dict[meta_item["business_id"]] = i

    with open(id2item_dir, "r") as f:
        datamaps = json.load(f)
    id2item = datamaps["id2item"]

    return meta_data, meta_dict, id2item


def find_metadata(index, meta_data, meta_dict, id2item):
    asin = id2item[index]  # index is string type
    meta_data_position = meta_dict[asin]
    index_meta_data = meta_data[meta_data_position]
    return index_meta_data


############### title-based representation ###############
# directly use title
def title_representation(index, meta_data, meta_dict, id2item):
    meta_information = find_metadata(index, meta_data, meta_dict, id2item)
    if "title" not in meta_information:
        title = "title_{}".format(index)
    else:
        title = meta_information["title"]
    if "name" in meta_information:
        title = meta_information["name"]
    return "[" + title + "]"


############### category-based representation ###############
def filter_categories(args, meta_data, meta_dict, id2item):
    indices = list(range(1, args.number_of_items))
    random.shuffle(indices)

    all_possible_categories = []

    for i in indices:
        index = str(i)
        categories = find_metadata(index, meta_data, meta_dict, id2item)["categories"]
        if categories is None:
            categories = ""
        categories = categories.split(", ")

        all_possible_categories += categories

    all_possible_categories = Counter(all_possible_categories)

    all_possible_categories = {
        k: v for k, v in all_possible_categories.items() if v > 10
    }

    return all_possible_categories


# content category-based ID
def build_category_map(args, meta_data, meta_dict, id2item):
    category_dict = {}
    category_counts = {}
    # randomize it to avoid data leakage
    # beauty 12102
    # toys 11925
    # sports 18358
    # yelp 20034
    indices = list(range(1, args.number_of_items))
    random.shuffle(indices)

    # to count in yelp time
    all_possible_categories = {}

    for i in indices:
        index = str(i)
        if args.task != "yelp":
            categories = find_metadata(index, meta_data, meta_dict, id2item)[
                "categories"
            ][0][1:]
        else:
            categories = find_metadata(index, meta_data, meta_dict, id2item)[
                "categories"
            ]
            if categories is None:
                categories = ""
            categories = categories.split(", ")

        # to count in yelp time
        for c in categories:
            if c not in all_possible_categories:
                all_possible_categories[c] = 1
            else:
                all_possible_categories[c] += 1
        # if no category
        if categories == [] or categories == [""]:
            categories = ["{}".format(i)]

        if args.task != "yelp":
            category_dict[index] = categories
            if tuple(categories) in category_counts:
                category_counts[tuple(categories)] += 1
            else:
                category_counts[tuple(categories)] = 1
            category_dict[index] = categories + [
                str(category_counts[tuple(categories)])
            ]

    if args.task == "yelp":
        filtered_categories = filter_categories(args, meta_data, meta_dict, id2item)
        for i in indices:
            index = str(i)
            # find categories in meta data
            categories = find_metadata(index, meta_data, meta_dict, id2item)[
                "categories"
            ]
            if categories is None:
                categories = ""
            categories = categories.split(", ")
            # filter categories
            categories = {
                c: filtered_categories[c]
                for c in categories
                if c in filtered_categories.keys()
            }
            # sort categories by order
            categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
            categories = [category[0] for category in categories][:5]
            # if no category
            if categories == [] or categories == [""]:
                categories = ["{}".format(i)]
            category_dict[index] = categories
            if tuple(categories) in category_counts:
                category_counts[tuple(categories)] += 1
            else:
                category_counts[tuple(categories)] = 1
            category_dict[index] = categories + [
                str(category_counts[tuple(categories)])
            ]

    max_subcategory = max([len(k) for k in list(category_counts.keys())])

    level_categories = []
    for i in range(max_subcategory):
        one_level_categories = set({})
        for categories in list(category_counts.keys()):
            if len(categories) > i:
                one_level_categories.add(categories[i])
        one_level_categories = sorted(one_level_categories)
        one_level_categories = {v: k for k, v in enumerate(list(one_level_categories))}
        level_categories.append(one_level_categories)

    return (category_dict, level_categories)


def content_based_representation(args, index, category_dict, level_categories):
    categories = category_dict[index]
    content_based_string = ""
    if args.task != "yelp":
        for i, c in enumerate(categories):
            if i + 1 != len(categories):
                # category_number = level_categories[i][c]
                # s = "<category_{}_{}>".format(i, category_number)
                s = "<category_{}>".format(c.replace(" ", "_"))
                content_based_string += s
            else:
                if not categories[0].isdigit():
                    token_number = categories[-1]
                    s = "<token_{}>".format(token_number)
                    # content_based_string += s
                    # content_based_string = str(token_number) + content_based_string
                    # content_based_string = content_based_string + str(token_number)
                    # s = "<token_{}>".format(index)
                    content_based_string += s
    else:
        for i, c in enumerate(categories):
            if i + 1 != len(categories):
                s = "<category_{}>".format(c.replace(" ", "_"))
                content_based_string += s
            else:
                if not categories[0].isdigit():
                    # token_number = categories[-1]
                    # s = "<token_{}>".format(token_number)
                    # content_based_string += s
                    s = "<token_{}>".format(index)
                    content_based_string += s
    return content_based_string


def build_category_map_modified_yelp(args):
    with open(args.data_dir + args.task + "/closed_categories.json", "r") as f:
        category_dict = json.load(f)

    category_counts = {}
    full_category_dict = {}
    for k, categories in category_dict.items():
        if tuple(categories) in category_counts:
            category_counts[tuple(categories)] += 1
        else:
            category_counts[tuple(categories)] = 1
        full_category_dict[k] = categories + [category_counts[tuple(categories)]]

    return full_category_dict


def content_based_representation_modified_yelp(index, category_dict):
    categories = category_dict[index]
    content_based_string = ""
    for i, c in enumerate(categories):
        if len(categories) == 1:
            content_based_string = "<category_{}_{}>".format(i, c)
        else:
            if i + 1 != len(categories):
                s = "<category_{}_{}>".format(i, c.replace(" ", "_"))
                content_based_string += s
            else:
                if not categories[0].isdigit():
                    token_number = categories[-1]
                    s = "<token_{}>".format(token_number)
                    # content_based_string += s
                    # s = "<token_{}>".format(index)
                    content_based_string += s
    return content_based_string


def create_category_embedding(args, tokenizer):
    if args.task == "beauty":
        categories = {
            "level_0": 6,
            "level_1": 38,
            "level_2": 148,
            "level_3": 51,
            "level_4": 8,
            # "max_token": 712,
            "max_token": 12101,
        }
    elif args.task == "toys":
        categories = {
            "level_0": 368,
            "level_1": 168,
            "level_2": 144,
            "level_3": 52,
            "level_4": 17,
            "level_5": 9,
            "level_6": 2,
            # "max_token": 822,
            "max_token": 11925,
        }
    else:
        assert args.task == "sports"
        categories = {
            "level_0": 279,
            "level_1": 163,
            "level_2": 515,
            "level_3": 503,
            "level_4": 105,
            "level_5": 4,
            # "max_token": 557,
            "max_token": 18358,
        }

    new_tokens = []
    for k, v in categories.items():
        if k.startswith("level"):
            level_number = k[-1]
            new_tokens += ["<category_{}_{}>".format(level_number, i) for i in range(v)]
        else:
            new_tokens += ["<token_{}>".format(i) for i in range(1, v + 1)]
    new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))

    return tokenizer


############### CF-based representation ###############
def CF_representation(x, mapping):
    x = str(int(x) - 1)
    if x in mapping:
        return mapping[x]
    else:
        return "<A{}>".format(x)


def create_CF_embedding(args, tokenizer):
    _, vocabulary = construct_indices_from_cluster(args)
    new_tokens = set(vocabulary) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))

    return tokenizer


def CF_representation_optimal_width(x, mapping):
    x = str(int(x) - 1)
    if x in mapping:
        return mapping[x]
    else:
        return "<A{}>".format(x)


def create_CF_embedding_optimal_width(args, tokenizer):
    _, vocabulary = construct_indices_from_cluster_optimal_width(args)
    new_tokens = set(vocabulary) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))

    return tokenizer


############### hybrid-based representation ###############
def index_mapping_computation(args):
    remapped_data_dir = args.data_dir + args.task + "/remapped_sequential_data.txt"

    with open(remapped_data_dir, "r") as f:
        remapped_data = f.read()
    remapped_data = remapped_data.split("\n")[:-1]
    remapped_items = []
    for item_list in remapped_data:
        remapped_items += item_list.split(" ")[1:]

    data_dir = args.data_dir + args.task + "/sequential_data.txt"
    with open(data_dir, "r") as f:
        data = f.read()
    data = data.split("\n")[:-1]
    items = []
    for item_list in data:
        items += item_list.split(" ")[1:]

    index_mapping = {}
    for original, remapped in tqdm(zip(items, remapped_items)):
        index_mapping[original] = remapped

    return index_mapping


def hybrid_dict_computation(
    args, category_dict, CF_dict, level_categories, index_mapping
):
    category_vocabulary = []
    hybrid_dict = {}
    for index in range(1, args.number_of_items):
        index = str(index)
        categories = category_dict[index]
        content_based_string = ""
        CF_based_string = ""
        for i, c in enumerate(categories):
            if i + 1 != len(categories):
                category_number = level_categories[i][c]
                s = "<category_{}_{}>".format(i, category_number)
                category_vocabulary.append(s)
                content_based_string += s
            else:
                cf_index = index_mapping[index]
                cf_index = str(int(cf_index) - 1)
                if cf_index in CF_dict:
                    s = CF_dict[cf_index]
                else:
                    s = "<A{}>".format(cf_index)
                CF_based_string = s

        # concatenate in different order
        if args.hybrid_order == "category_first":
            hybrid_string = content_based_string + CF_based_string
        else:
            assert args.hybrid_order == "CF_first"
            hybrid_string = CF_based_string + content_based_string
        hybrid_dict[index] = hybrid_string

    return hybrid_dict, category_vocabulary


def load_hybrid(args):
    meta_dir = args.data_dir + args.task + "/meta.json.gz"
    id2item_dir = args.data_dir + args.task + "/datamaps.json"

    def parse(path):
        g = gzip.open(path, "r")
        for l in g:
            yield eval(l)

    meta_data = []
    for meta in parse(meta_dir):
        meta_data.append(meta)

    meta_dict = {}
    for i, meta_item in enumerate(meta_data):
        meta_dict[meta_item["asin"]] = i

    with open(id2item_dir, "r") as f:
        datamaps = json.load(f)
    id2item = datamaps["id2item"]

    category_dict, level_categories = build_category_map(
        args, meta_data, meta_dict, id2item
    )

    if args.optimal_width_in_CF:
        CF_dict, CF_vocabulary = construct_indices_from_cluster_optimal_width(args)
    else:
        CF_dict, CF_vocabulary = construct_indices_from_cluster(args)

    index_mapping = index_mapping_computation(args)

    hybrid_dict, category_vocabulary = hybrid_dict_computation(
        args, category_dict, CF_dict, level_categories, index_mapping
    )

    vocabulary = category_vocabulary + CF_vocabulary

    return hybrid_dict, vocabulary


def hybrid_representation(index, hybrid_dict):
    return hybrid_dict[index]


def create_hybrid_embedding(vocabulary, tokenizer):
    new_tokens = set(vocabulary) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))

    return tokenizer


if __name__ == "__main__":

    random.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)
    torch.cuda.manual_seed_all(2022)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--task", type=str, default="beauty")
    parser.add_argument("--cluster_size", type=int, default=100)
    parser.add_argument("--cluster_number", type=int, default=20)

    parser.add_argument("--category_no_repetition", action="store_true")
    parser.add_argument(
        "--number_of_items",
        type=int,
        default=11925,
        help="number of items in each dataset, beauty 12102, toys 11925, sports 18358",
    )

    parser.add_argument(
        "--hybrid_order",
        type=str,
        default="CF_first",
        help="CF_first or category_first in concatenation",
    )

    args = parser.parse_args()

    """
    random_fct = random_number(30000)
    resolution = 4
    overlap = 0
    base = 10
    remap_fct = lambda x: item_resolution(resolution=resolution, overlap=overlap)(
        change_base(random_fct(x), base)
    )
    # remap_fct = lambda x: item_resolution(resolution=resolution, overlap=overlap)(
    #    random_fct(x)
    # )

    print(remap_fct("1"))
    print(remap_fct("2"))
    print(remap_fct("3"))
    print(remap_fct("4"))
    print(remap_fct("5"))
    print(remap_fct("6"))
    print(remap_fct("7"))
    print(remap_fct("8"))
    print(remap_fct("9"))
    print(remap_fct("10"))
    print(remap_fct("11"))
    print(remap_fct("12"))
    print(remap_fct("13"))
    print(remap_fct("14"))
    print(remap_fct("15"))
    print(remap_fct("16"))
    """

    if args.task != "yelp":
        meta_dir = "data/{}/meta.json.gz".format(args.task)
        id2item_dir = "data/{}/datamaps.json".format(args.task)
    else:
        meta_dir = "data/yelp/meta_data.pkl"
        id2item_dir = "data/yelp/datamaps.json"

    id2item_dir = args.data_dir + args.task + "/datamaps.json"

    def parse(path):
        g = gzip.open(path, "r")
        for l in g:
            yield eval(l)

    if args.task != "yelp":
        meta_data = []
        for meta in parse(meta_dir):
            meta_data.append(meta)

        meta_dict = {}
        for i, meta_item in enumerate(meta_data):
            meta_dict[meta_item["asin"]] = i
    else:
        with open(meta_dir, "rb") as f:
            meta_data = pkl.load(f)
        meta_dict = {}
        for i, meta_item in enumerate(meta_data):
            meta_dict[meta_item["business_id"]] = i

    with open(id2item_dir, "r") as f:
        datamaps = json.load(f)
    id2item = datamaps["id2item"]

    category_dict, level_categories = build_category_map(
        args, meta_data, meta_dict, id2item
    )

    reps = []
    for i in tqdm(range(1, args.number_of_items)):
        index = str(i)
        rep = content_based_representation(args, index, category_dict, level_categories)
        print(rep)
        reps.append(rep)
        time.sleep(5)
    print(len(reps))
    repsc = Counter(reps)
    for k, v in repsc.items():
        if v > 1:
            print((k, v))
    print(len(set(reps)))

    # mapping, vocabulary = construct_indices_from_cluster(args)
    # for index in tqdm(range(11800, 12102)):
    #    index = str(index)
    #    # rep = title_representation(index, meta_data, meta_dict, id2item)
    #    rep = CF_representation(index, mapping)
    #    print(rep)
    #    time.sleep(1)

    """
    hybrid_dict, vocabulary = load_hybrid(args)

    for vocab in vocabulary:
        if "new_token" in vocab:
            print(vocab)
    print(len(hybrid_dict))
    reps = list(hybrid_dict.values())
    print(len(reps) == len(list(set(reps))))
    print(hybrid_dict["1"])
    print(hybrid_dict["2"])
    print(hybrid_dict["3"])
    print(hybrid_dict["4"])
    print(hybrid_dict["5"])
    print(hybrid_dict["12068"])
    print(hybrid_dict["12069"])
    print(hybrid_dict["12070"])
    # for k, v in tqdm(content_based_representation_dict.items()):
    #    print((k, v))
    #    print("***")
    #    time.sleep(3)
    # print(len(category_counts))
    # for k, v in category_counts.items():
    #    if v > 100:
    #        print((k, v))
    #        print("***")
    """

    """
    index = "21"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "22"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "23"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "24"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "25"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "26"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "27"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "28"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "29"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "10"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "11"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "12"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "13"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "14"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "15"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "16"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "17"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "18"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "19"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)

    index = "20"
    rep = content_based_representation(index, category_dict, level_categories)
    print(category_dict[index])
    print(rep)
    """
