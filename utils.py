import torch
import random
import numpy as np
import os
import sys
from datetime import date, datetime
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import OrderedDict
import torch.nn as nn
import json
import gzip
import time
import pickle as pkl
from sklearn.metrics import ndcg_score


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


class Logger(object):
    def __init__(self, log_path, on=True):
        self.log_path = log_path
        self.on = on
        if self.on:
            while os.path.isfile(self.log_path):
                self.log_path += "+"

    def log(self, string, newline=True):
        if self.on:
            with open(self.log_path, "a") as logf:
                today = date.today()
                today_date = today.strftime("%m/%d/%Y")
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                string = today_date + ", " + current_time + ": " + string
                logf.write(string)
                if newline:
                    logf.write("\n")

            sys.stdout.write(string)
            if newline:
                sys.stdout.write("\n")
            sys.stdout.flush()


def create_optimizer_and_scheduler(
    args, logger, model, batch_per_epoch, pretrain="None"
):
    if pretrain == "meta":
        total_steps = (
            batch_per_epoch // args.gradient_accumulation_steps * args.meta_epochs
        )
    elif pretrain == "review":
        total_steps = (
            batch_per_epoch // args.gradient_accumulation_steps * args.review_epochs
        )
    else:
        total_steps = batch_per_epoch // args.gradient_accumulation_steps * args.epochs
    warmup_steps = int(total_steps * args.warmup_prop)

    if args.gpu == 0:
        logger.log("Batch per epoch: %d" % batch_per_epoch)
        logger.log("Total steps: %d" % total_steps)
        logger.log("Warmup proportion:", args.warmup_prop)
        logger.log("Warm up steps: %d" % warmup_steps)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if pretrain == "meta":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.meta_lr, eps=args.adam_eps
        )
    elif pretrain == "review":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.review_lr, eps=args.adam_eps
        )
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    return optimizer, scheduler


def load_model(model, pretrained_dir, rank):
    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    ckpt = torch.load(pretrained_dir, map_location=map_location)
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        # k = k.replace("module_", "")
        k = k.replace("module.", "")
        # k = "module_" + k
        new_ckpt[k] = v

    model.load_state_dict(new_ckpt, strict=True)

    return model


def remove_extra_id(x):
    x = x.replace("<extra_id_", "")
    x = x.replace(">", "")
    return x


def exact_match(predictions, targets, scores, total_sequence_generated):
    batched_predictions = []
    batched_scores = []
    batch_length = len(targets)
    for b in range(batch_length):
        one_batch_sequence = predictions[
            b * total_sequence_generated : (b + 1) * total_sequence_generated
        ]
        one_batch_score = scores[
            b * total_sequence_generated : (b + 1) * total_sequence_generated
        ]
        pairs = [(a, b) for a, b in zip(one_batch_sequence, one_batch_score)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        batched_predictions.append([sorted_pair[0] for sorted_pair in sorted_pairs])
        batched_scores.append([float(sorted_pair[1]) for sorted_pair in sorted_pairs])
    hit_1 = 0
    hit_5 = 0
    hit_10 = 0
    ncdg_5 = 0
    ncdg_10 = 0
    for p, s, t in zip(batched_predictions, batched_scores, targets):
        # hit@k
        if remove_extra_id(t) in [remove_extra_id(one_p) for one_p in p[:1]]:
            hit_1 += 1
        if remove_extra_id(t) in [remove_extra_id(one_p) for one_p in p[:5]]:
            hit_5 += 1
        if remove_extra_id(t) in [remove_extra_id(one_p) for one_p in p[:10]]:
            hit_10 += 1
        # ncdg@k
        if remove_extra_id(t) in [remove_extra_id(one_p) for one_p in p[:5]]:
            gold_position = [remove_extra_id(one_p) for one_p in p[:5]].index(
                remove_extra_id(t)
            )
            true_scores = [0.0] * 5
            true_scores[gold_position] = 1.0
            true_scores_5 = np.array([true_scores])
            predict_scores_5 = np.array([s[:5]])
            ncdg_5 += ndcg_score(true_scores_5, predict_scores_5)
        else:
            ncdg_5 += 0
        if remove_extra_id(t) in [remove_extra_id(one_p) for one_p in p[:10]]:
            gold_position = [remove_extra_id(one_p) for one_p in p[:10]].index(
                remove_extra_id(t)
            )
            true_scores = [0.0] * 10
            true_scores[gold_position] = 1.0
            true_scores_10 = np.array([true_scores])
            predict_scores_10 = np.array([s[:10]])
            ncdg_10 += ndcg_score(true_scores_10, predict_scores_10)
        else:
            ncdg_10 += 0

    return hit_1, hit_5, hit_10, ncdg_5, ncdg_10


def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens


def random_initialization(model, tokenizer):
    ids = []
    for x in range(50000):
        tokenized_ids = tokenizer.encode(str(x))
        if 3 in tokenized_ids:
            tokenized_ids.remove(3)
        if 1 in tokenized_ids:
            tokenized_ids.remove(1)
        ids += tokenized_ids
        tokenized_ids = tokenizer.encode("item_" + str(x))
        if 3 in tokenized_ids:
            tokenized_ids.remove(3)
        if 1 in tokenized_ids:
            tokenized_ids.remove(1)
        tokenized_ids.remove(2118)
        tokenized_ids.remove(834)
        ids += tokenized_ids
    ids = list(set(ids))
    for index in ids:
        model.shared.weight.data[index] = nn.init.uniform_(
            model.shared.weight.data[index], -10.0, 10.0
        )

    return model


def add_new_tokens(model, tokenizer):
    new_tokens = []
    for x in range(12101):
        new_token = "<extra_id_{}>".format(x)
        new_tokens.append(new_token)
    new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))

    # add new, random embeddings for the new tokens
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


"""
def create_category_embedding(tokenizer):
    categories = {
        "level_0": 6,
        "level_1": 38,
        "level_2": 148,
        "level_3": 51,
        "level_4": 8,
        # "max_token": 712,
        "max_token": 12101,
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
"""


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


def content_based_representation_non_hierarchical(
    args, category_dict, level_categories, tokenizer
):
    new_tokens = []
    for i in range(1, args.number_of_items):
        index = str(i)
        categories = category_dict[index]
        for i, c in enumerate(categories):
            if i + 1 != len(categories):
                # category_number = level_categories[i][c]
                # s = "<category_{}_{}>".format(i, category_number)
                s = "<category_{}>".format(c.replace(" ", "_"))
                new_tokens.append(s)
            else:
                if not categories[0].isdigit():
                    token_number = categories[-1]
                    s = "<token_{}>".format(token_number)
                    # content_based_string += s
                    # content_based_string = str(token_number) + content_based_string
                    # content_based_string = content_based_string + str(token_number)
                    # s = "<token_{}>".format(index)
                    new_tokens.append(s)

    new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))

    return tokenizer


def create_category_embedding_yelp(args, category_dict, level_categories, tokenizer):
    assert args.task == "yelp"
    new_tokens = []
    for i in range(1, args.number_of_items):
        index = str(i)
        categories = category_dict[index]
        for i, c in enumerate(categories):
            if i + 1 != len(categories):
                s = "<category_{}>".format(c.replace(" ", "_"))
                new_tokens.append(s)
            else:
                if not categories[0].isdigit():
                    # token_number = categories[-1]
                    # s = "<token_{}>".format(token_number)
                    s = "<token_{}>".format(index)
                    new_tokens.append(s)

    new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))

    return tokenizer


def content_category_embedding_modified_yelp(args, category_dict, tokenizer):
    assert args.task == "yelp"
    new_tokens = []
    for i in range(1, args.number_of_items):
        index = str(i)
        categories = category_dict[index]
        for i, c in enumerate(categories):
            if len(categories) == 1:
                s = "<category_{}_{}>".format(i, c)
                new_tokens.append(s)
            else:
                if i + 1 != len(categories):
                    s = "<category_{}_{}>".format(i, c.replace(" ", "_"))
                    new_tokens.append(s)
                else:
                    if not categories[0].isdigit():
                        token_number = categories[-1]
                        s = "<token_{}>".format(token_number)
                        # s = "<token_{}>".format(index)
                        new_tokens.append(s)

    new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))

    return tokenizer


def save_review(args):
    data_dir = args.data_dir + args.task + "/review_splits_augmented.pkl"
    review_file = open(data_dir, "rb")
    reviews = pkl.load(review_file)

    id2item_dir = args.data_dir + args.task + "/datamaps.json"
    with open(id2item_dir, "r") as f:
        datamaps = json.load(f)
    item2id = datamaps["item2id"]
    user2id = datamaps["user2id"]

    cleaned_reviews = []
    for review in reviews:
        reviwerID = review["reviewerID"]
        itemID = review["asin"]
        text = review["reviewText"]
        cleaned_reviews.append(
            {"user": user2id[reviwerID], "item": item2id[itemID], "text": text}
        )

    with open(args.data_dir + args.task + "/sequential_data.txt", "r") as f:
        sequential = f.read()
    sequential = sequential.split("\n")
    sequential = [s.split(" ") for s in sequential]
    sequential = {s[0]: s[1:-2] for s in sequential}
    pairs = []
    for k, v in sequential.items():
        for i in v:
            pairs.append((k, i))

    train_reviews = []
    for r in cleaned_reviews:
        if (r["user"], r["item"]) in pairs:
            train_reviews.append(r)

    with open(args.data_dir + args.task + "/train_reviews.json", "w") as f:
        json.dump(train_reviews, f)

    return cleaned_reviews


if __name__ == "__main__":
    """
    from transformers import T5Model, T5Tokenizer

    model = T5Model.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = random_initialization(model, tokenizer)

    print(model.shared.weight.data[204])
    """

    # meta_data, meta_dict, id2item = load_meta()

    from transformers import T5Model, AutoTokenizer

    model = T5Model.from_pretrained("t5-small")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    create_category_embedding(tokenizer)
