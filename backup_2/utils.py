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


def create_optimizer_and_scheduler(args, logger, model, batch_per_epoch):
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

    model.load_state_dict(new_ckpt)

    return model


def exact_match(predictions, targets, k):
    batched_predictions = []
    batch_length = len(targets)
    for b in range(batch_length):
        batched_predictions.append(predictions[b * k : (b + 1) * k])
    correct = 0
    for p, t in zip(batched_predictions, targets):
        if t in p:
            correct += 1

    return correct


def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens


def random_initialization(model, tokenizer):
    ids = []
    for x in range(30000):
        tokenized_ids = tokenizer.encode(str(x))
        if 3 in tokenized_ids:
            tokenized_ids.remove(3)
        if 1 in tokenized_ids:
            tokenized_ids.remove(1)
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


def load_meta(args):
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

    return meta_data, meta_dict, id2item


def find_metadata(index, meta_data, meta_dict, id2item):
    asin = id2item[index]  # index is string type
    meta_data_position = meta_dict[asin]
    index_meta_data = meta_data[meta_data_position]
    return index_meta_data


if __name__ == "__main__":
    """
    from transformers import T5Model, T5Tokenizer

    model = T5Model.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = random_initialization(model, tokenizer)

    print(model.shared.weight.data[204])
    """

    meta_data, meta_dict, id2item = load_meta()
