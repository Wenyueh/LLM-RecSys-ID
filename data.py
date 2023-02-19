import random
import argparse
from re import template
from torch.nn import modules
from torch.utils.data import Dataset, DataLoader
from prompt import (
    task_subgroup_1,
    task_subgroup_2,
    task_subgroup_3,
    task_subgroup_4,
    task_subgroup_5,
    task_subgroup_6,
    task_subgroup_8,
    task_subgroup_9,
)
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import math
from torch.utils.data.distributed import DistributedSampler
from item_rep_method import (
    random_number,
    random_one_token,
    random_two_token,
    no_tokenization,
)
from utils import load_meta, find_metadata
import time
import json
import numpy as np


def load_data(args, tokenizer):
    data_dir = args.data_dir + args.task + "/sequential_data.txt"
    with open(data_dir, "r") as f:
        data = f.read()
    data = data.split("\n")[:-1]

    users = []
    all_items = []
    remapped_all_items = []
    train_sequence = []
    val_sequence = []
    test_sequence = []

    remap_fct = None
    if args.item_representation == "random_number":
        remap_fct = random_number(30000)
    if args.item_representation == "random_one_token":
        remap_fct = random_one_token(tokenizer)
    if args.no_tokenization:
        remap_fct = no_tokenization()

    for one_user in data:
        splittion_point = one_user.index(" ")
        user = one_user[:splittion_point]
        items = one_user[splittion_point + 1 :].split(" ")
        if (
            args.item_representation in ["random_number", "random_one_token"]
            or args.no_tokenization
        ):
            remapped_items = [remap_fct(item) for item in items]
        else:
            remapped_items = items

        users.append(user)
        train_sequence.append(remapped_items[:-2])
        val_sequence.append(remapped_items[:-1])
        test_sequence.append(remapped_items)
        all_items += items
        remapped_all_items += remapped_items

    remove_extra_items = list(
        set([(a, b) for a, b in zip(all_items, remapped_all_items)])
    )
    remove_extra_items = sorted(remove_extra_items, key=lambda x: x[0])
    all_items = [pair[0] for pair in remove_extra_items]
    remapped_all_items = [pair[1] for pair in remove_extra_items]

    return (
        users,
        all_items,
        train_sequence,
        val_sequence,
        test_sequence,
        remapped_all_items,
    )


class sequential_item_dataset(Dataset):
    def __init__(self, args, users, train_sequence, task_group):
        super().__init__()
        self.args = args
        self.users = users
        self.train_history = train_sequence
        self.templates = task_group
        self.num_template = len(self.templates)
        self.number_of_interactions()

    def number_of_interactions(self):
        self.user_interaction_code = []
        total_num = 0
        for k, v in zip(self.users, self.train_history):
            number = len(v[1:])
            total_num += number
            for _ in range(number):
                self.user_interaction_code.append(k)
        return total_num

    def __len__(self):
        number = self.number_of_interactions() * self.num_template

        return number

    def __getitem__(self, index):

        position_index = index // self.num_template

        user_idx = self.user_interaction_code[position_index]
        the_number_of_datapoint = self.users.index(user_idx)
        whole_sequence = self.train_history[the_number_of_datapoint]

        # target
        target_item_index = random.choice(range(1, len(whole_sequence)))
        target_item = whole_sequence[target_item_index]

        # history
        purchase_history = whole_sequence[:target_item_index]

        if len(purchase_history) > 20:
            purchase_history = purchase_history[-20:]

        template_idx = index % self.num_template
        template = self.templates[template_idx]

        if template["input_first"] == "user":
            input_sent = template["source"].format(
                user_idx,
                " , ".join(["item_" + item_idx for item_idx in purchase_history]),
            )
        else:
            input_sent = template["source"].format(
                " , ".join(["item_" + item_idx for item_idx in purchase_history]),
                user_idx,
            )
        output_sent = template["target"].format("item_" + target_item)

        return input_sent, output_sent


class sequential_yesno_dataset(Dataset):
    def __init__(self, args, users, all_items, train_history, task_group):
        super().__init__()
        self.args = args
        self.users = users
        self.all_items = all_items
        self.train_history = train_history
        self.templates = task_group
        self.num_template = len(self.templates)

    def __len__(self):
        number = (
            len(self.users)
            * (self.args.negative_sample + 1)
            * self.num_template
            * self.args.sequential_num
        )
        return number

    def __getitem__(self, index):
        polarity = "pos" if index % (1 + self.args.negative_sample) == 0 else "neg"
        index = index // (1 + self.args.negative_sample)
        # each user record is splitted into sequential_num many sequences
        non_random_index = int(index / self.args.sequential_num)
        the_number_of_datapoint = non_random_index // self.num_template

        user_idx = self.users[the_number_of_datapoint]
        sequence = self.train_history[the_number_of_datapoint]
        end_candidates = [
            _
            for _ in range(
                max(2, len(sequence) - self.args.sequential_num - 1), len(sequence) - 1,
            )
        ]
        end_index = random.randint(0, len(end_candidates) - 1)
        end_pos = end_candidates[end_index]
        start_candidates = [_ for _ in range(1, min(4, end_pos))]
        start_index = random.randint(0, len(start_candidates) - 1)
        start_pos = start_candidates[start_index]
        purchase_history = sequence[start_pos : end_pos + 1]
        if len(purchase_history) > self.args.max_history:
            purchase_history = purchase_history[-self.args.max_history :]
        target_item = sequence[end_pos + 1]

        template_idx = non_random_index % self.num_template
        template = self.templates[template_idx]

        if template["input_first"] == "user":
            if polarity == "pos":
                input_sent = template["source"].format(
                    user_idx,
                    " , ".join(["item_" + item_idx for item_idx in purchase_history]),
                    "item_" + target_item,
                )
                output_sent = template["target"].format("yes")
            else:
                candidates = self.all_items.copy()
                candidates.remove(target_item)
                negative_item = random.choice(candidates)
                input_sent = template["source"].format(
                    user_idx,
                    " , ".join(["item_" + item_idx for item_idx in purchase_history]),
                    "item_" + negative_item,
                )
                output_sent = template["target"].format("no")
        else:
            if polarity == "pos":
                input_sent = template["source"].format(
                    " , ".join(["item_" + item_idx for item_idx in purchase_history]),
                    user_idx,
                    "item_" + target_item,
                )
                output_sent = template["target"].format("yes")
            else:
                candidates = self.all_items.copy()
                candidates.remove(target_item)
                negative_item = random.choice(candidates)
                input_sent = template["source"].format(
                    " , ".join(["item_" + item_idx for item_idx in purchase_history]),
                    user_idx,
                    "item_" + negative_item,
                )
                output_sent = template["target"].format("no")

        return input_sent, output_sent


class direct_yesno_dataset(Dataset):
    def __init__(self, args, users, all_items, test_history, task_group):
        super().__init__()
        self.args = args
        self.users = users
        self.all_items = all_items
        self.all_history = test_history
        self.templates = task_group
        self.num_template = len(self.templates)

    def __len__(self):
        number = (
            len(self.users)
            * (self.args.negative_sample + 1)
            * self.num_template
            * self.args.yes_no_sample
        )
        return number

    def __getitem__(self, index):
        # for each user, we do yes_no_sample many times of direct yes/no question
        index = int(index / self.args.yes_no_sample)

        polarity = "pos" if index % (1 + self.args.negative_sample) == 0 else "neg"
        index = index // (1 + self.args.negative_sample)

        the_number_of_datapoint = index // self.num_template
        user_idx = self.users[the_number_of_datapoint]
        sequence = self.all_history[the_number_of_datapoint]

        template_idx = index % self.num_template
        template = self.templates[template_idx]

        if template["input_first"] == "user":
            if polarity == "pos":
                target_item = random.choice(sequence[:-2])
                input_sent = template["source"].format(user_idx, target_item)
                output_sent = template["target"].format("yes")
            else:
                negative_items = self.all_items.copy()
                negative_items = list(set(negative_items) - set(sequence))
                negative_item = random.choice(negative_items)
                input_sent = template["source"].format(user_idx, negative_item)
                output_sent = template["target"].format("no")
        else:
            if polarity == "pos":
                target_item = random.choice(sequence[:-2])
                input_sent = template["source"].format(target_item, user_idx)
                output_sent = template["target"].format("yes")
            else:
                negative_items = self.all_items.copy()
                negative_items = list(set(negative_items) - set(sequence))
                negative_item = random.choice(negative_items)
                input_sent = template["source"].format(negative_item, user_idx)
                output_sent = template["target"].format("no")

        return input_sent, output_sent


class direct_candidate_dataset(Dataset):
    def __init__(self, args, users, all_items, test_history, task_group):
        super().__init__()
        self.args = args
        self.users = users
        self.all_items = all_items
        self.all_history = test_history
        self.templates = task_group
        self.num_template = len(self.templates)
        self.train_history = [d[:-2] for d in self.all_history]
        self.number_of_interactions()

    def number_of_interactions(self):
        self.user_interaction_code = []
        total_num = 0
        for k, v in zip(self.users, self.train_history):
            number = len(v)
            total_num += number
            for _ in range(number):
                self.user_interaction_code.append(k)
        return total_num

    def __len__(self):
        length = self.number_of_interactions() * self.num_template
        return length

    def __getitem__(self, index):
        index = index // self.num_template

        user_idx = self.user_interaction_code[index]
        the_number_of_datapoint = self.users.index(user_idx)
        sequence = self.all_history[the_number_of_datapoint]

        target_item = random.choice(sequence[:-2])
        negative_items = self.all_items.copy()
        negative_items = list(set(negative_items) - set(sequence))
        negative_items = random.sample(negative_items, k=100)
        candidates = [target_item] + negative_items
        random.shuffle(candidates)

        template_idx = index % self.num_template
        template = self.templates[template_idx]

        if template["input_first"] == "user":
            input_sent = template["source"].format(
                user_idx, "items " + " , ".join(candidates)
            )
            output_sent = template["target"].format("item_" + target_item)
        else:
            input_sent = template["source"].format(
                "items " + " , ".join(candidates), user_idx
            )
            output_sent = template["target"].format("item_" + target_item)

        return input_sent, output_sent


class direct_straightforward_dataset(Dataset):
    def __init__(self, args, users, all_items, test_history, task_group):
        super().__init__()
        self.args = args
        self.users = users
        self.all_items = all_items
        self.all_history = test_history
        self.templates = task_group
        self.num_template = len(self.templates)
        self.train_history = [d[:-2] for d in self.all_history]
        self.number_of_interactions()

    def number_of_interactions(self):
        self.user_interaction_code = []
        total_num = 0
        for k, v in zip(self.users, self.train_history):
            number = len(v)
            total_num += number
            for _ in range(number):
                self.user_interaction_code.append(k)
        return total_num

    def __len__(self):
        length = self.number_of_interactions() * self.num_template
        return length

    def __getitem__(self, index):
        index = index // self.num_template

        user_idx = self.user_interaction_code[index]
        the_number_of_datapoint = self.users.index(user_idx)
        sequence = self.all_history[the_number_of_datapoint]

        target_item = random.choice(sequence[:-2])

        template_idx = index % self.num_template
        template = self.templates[template_idx]

        if template["input_first"] == "user":
            input_sent = template["source"].format(user_idx)
            output_sent = template["target"].format("item_" + target_item)
        else:
            input_sent = template["source"].format(user_idx)
            output_sent = template["target"].format("item_" + target_item)

        return input_sent, output_sent


def calculate_whole_word_ids(tokenized_text, input_ids):
    whole_word_ids = []
    curr = 0
    for i in range(len(tokenized_text)):
        if tokenized_text[i] == "<pad>":
            curr = 0
        if tokenized_text[i].startswith("▁"):
            curr += 1
            whole_word_ids.append(curr)
        else:
            whole_word_ids.append(curr)
    return whole_word_ids[: len(input_ids) - 1] + [0]  # [0] for </s>


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_texts = [input_text[0] for input_text in batch]
        output_texts = [input_text[1] for input_text in batch]

        inputs = self.tokenizer.batch_encode_plus(
            input_texts, padding="longest", truncation=True, max_length=512
        )
        input_ids = inputs["input_ids"]
        whole_word_ids = []
        for input_id in input_ids:
            tokenized_text = self.tokenizer.convert_ids_to_tokens(input_id)
            whole_word_id = calculate_whole_word_ids(tokenized_text, input_id)
            whole_word_ids.append(whole_word_id)
        input_attention = inputs["attention_mask"]
        outputs = self.tokenizer.batch_encode_plus(
            output_texts, padding="longest", truncation=True, max_length=512
        )
        output_ids = outputs["input_ids"]
        output_attention = outputs["attention_mask"]

        return (
            torch.tensor(input_ids),
            torch.tensor(input_attention),
            torch.tensor(whole_word_ids),
            torch.tensor(output_ids),
            torch.tensor(output_attention),
        )


def load_train_dataloaders(
    args, tokenizer, users, all_items, train_sequence, test_sequence
):
    # load data
    # users, all_items, train_sequence, val_sequence, test_sequence = load_data(
    #    args, tokenizer
    # )

    task_data_lengths = []
    # create sequential item dataset
    train_sequential_item_dataset = sequential_item_dataset(
        args, users, train_sequence, task_subgroup_1
    )

    train_sequential_item_dataset_length = math.ceil(
        len(train_sequential_item_dataset) / args.train_sequential_item_batch
    )
    task_data_lengths.append(train_sequential_item_dataset_length)

    # create sequential yesno dataset
    train_sequential_yesno_dataset = sequential_yesno_dataset(
        args, users, all_items, train_sequence, task_subgroup_2
    )
    train_sequential_yesno_dataset_length = math.ceil(
        len(train_sequential_yesno_dataset) / args.train_sequential_yesno_batch
    )
    task_data_lengths.append(train_sequential_yesno_dataset_length)

    # create direct yesno dataset
    train_direct_yesno_dataset = direct_yesno_dataset(
        args, users, all_items, test_sequence, task_subgroup_3
    )
    train_direct_yesno_dataset_length = math.ceil(
        len(train_direct_yesno_dataset) / args.train_direct_yesno_batch
    )
    task_data_lengths.append(train_direct_yesno_dataset_length)

    # create direct candidate dataset
    train_direct_candidate_dataset = direct_candidate_dataset(
        args, users, all_items, test_sequence, task_subgroup_4
    )
    train_direct_candidate_dataset_length = math.ceil(
        len(train_direct_candidate_dataset) / args.train_direct_candidate_batch
    )
    task_data_lengths.append(train_direct_candidate_dataset_length)

    # create direct straightforward dataset
    train_direct_straightforward_dataset = direct_straightforward_dataset(
        args, users, all_items, test_sequence, task_subgroup_5
    )
    train_direct_straightforward_dataset_length = math.ceil(
        len(train_direct_straightforward_dataset)
        / args.train_direct_straightforward_batch
    )
    task_data_lengths.append(train_direct_straightforward_dataset_length)

    # create collator
    collator = Collator(tokenizer)

    # create sampler
    if args.distributed:
        sampler = DistributedSampler(train_sequential_item_dataset)
    else:
        sampler = None
    train_sequential_item_dataloader = DataLoader(
        train_sequential_item_dataset,
        batch_size=args.train_sequential_item_batch,
        collate_fn=collator,
        shuffle=(sampler is None),
        sampler=sampler,
    )

    # create sampler
    if args.distributed:
        sampler = DistributedSampler(train_sequential_yesno_dataset)
    else:
        sampler = None
    train_sequential_yesno_dataloader = DataLoader(
        train_sequential_yesno_dataset,
        batch_size=args.train_sequential_yesno_batch,
        collate_fn=collator,
        shuffle=(sampler is None),
        sampler=sampler,
    )

    # create sampler
    if args.distributed:
        sampler = DistributedSampler(train_direct_yesno_dataset)
    else:
        sampler = None
    train_direct_yesno_dataloader = DataLoader(
        train_direct_yesno_dataset,
        batch_size=args.train_direct_yesno_batch,
        collate_fn=collator,
        shuffle=(sampler is None),
        sampler=sampler,
    )

    # create sampler
    if args.distributed:
        sampler = DistributedSampler(train_direct_candidate_dataset)
    else:
        sampler = None
    train_direct_candidate_dataloader = DataLoader(
        train_direct_candidate_dataset,
        batch_size=args.train_direct_candidate_batch,
        collate_fn=collator,
        shuffle=(sampler is None),
        sampler=sampler,
    )

    # create sampler
    if args.distributed:
        sampler = DistributedSampler(train_direct_straightforward_dataset)
    else:
        sampler = None
    train_direct_straightforward_dataloader = DataLoader(
        train_direct_straightforward_dataset,
        batch_size=args.train_direct_straightforward_batch,
        collate_fn=collator,
        shuffle=(sampler is None),
        sampler=sampler,
    )

    return (
        train_sequential_item_dataloader,
        train_sequential_yesno_dataloader,
        train_direct_yesno_dataloader,
        train_direct_candidate_dataloader,
        train_direct_straightforward_dataloader,
        task_data_lengths,
        all_items,
    )


##################################################
##################################################
############### evaluation dataset ###############
##################################################
##################################################


class evaluation_sequential_item_dataset(Dataset):
    def __init__(self, args, users, eval_sequence, task_group, mode):
        super().__init__()
        self.args = args
        self.mode = mode
        self.evaluation_template_id = self.args.evaluation_template_id
        self.users = users
        self.eval_history = eval_sequence
        self.template = task_group[self.evaluation_template_id]

    def __len__(self):
        number = len(self.users)
        return number

    def __getitem__(self, index):
        user_idx = self.users[index]
        sequence = self.eval_history[index]
        if self.args.remove_last_item:
            if self.mode == "validation":
                purchase_history = sequence[:-1]
            else:
                assert self.mode == "test"
                purchase_history = sequence[:-2]
        elif self.args.remove_first_item:
            if self.mode == "validation":
                purchase_history = sequence[:-1]
            else:
                assert self.mode == "test"
                purchase_history = sequence[1:-1]
        else:
            purchase_history = sequence[:-1]
        if len(purchase_history) > self.args.max_history:
            purchase_history = purchase_history[-self.args.max_history :]
        target_item = sequence[-1]

        if self.template["input_first"] == "user":
            input_sent = self.template["source"].format(
                user_idx,
                " , ".join(["item_" + item_idx for item_idx in purchase_history]),
            )
        else:
            input_sent = self.template["source"].format(
                " , ".join(["item_" + item_idx for item_idx in purchase_history]),
                user_idx,
            )
        output_sent = self.template["target"].format("item_" + target_item)

        return input_sent, output_sent


class evaluation_sequential_yesno_dataset(Dataset):
    def __init__(self, args, users, all_items, eval_sequence, task_group):
        super().__init__()
        self.args = args
        self.evaluation_template_id = self.args.evaluation_template_id
        self.users = users
        self.all_items = all_items
        self.eval_history = eval_sequence
        self.template = task_group[self.evaluation_template_id]

    def __len__(self):
        number = len(self.users) * (100 + 1)
        return number

    def __getitem__(self, index):
        polarity = "pos" if index % (1 + 100) == 0 else "neg"
        index = index // (1 + 100)

        user_idx = self.users[index]
        sequence = self.eval_history[index]
        purchase_history = sequence[:-1]
        if len(purchase_history) > self.args.max_history:
            purchase_history = purchase_history[-self.args.max_history :]
        target_item = sequence[-1]

        if self.template["input_first"] == "user":
            if polarity == "pos":
                input_sent = self.template["source"].format(
                    user_idx,
                    " , ".join(["item_" + item_idx for item_idx in purchase_history]),
                    "item_" + target_item,
                )
                output_sent = self.template["target"].format("yes")
            else:
                candidates = self.all_items.copy()
                candidates.remove(target_item)
                negative_item = random.choice(candidates)
                input_sent = self.template["source"].format(
                    user_idx,
                    " , ".join(["item_" + item_idx for item_idx in purchase_history]),
                    "item_" + negative_item,
                )
                output_sent = self.template["target"].format("no")
        else:
            if polarity == "pos":
                input_sent = self.template["source"].format(
                    " , ".join(["item_" + item_idx for item_idx in purchase_history]),
                    user_idx,
                    "item_" + target_item,
                )
                output_sent = self.template["target"].format("yes")
            else:
                candidates = self.all_items.copy()
                candidates.remove(target_item)
                negative_item = random.choice(candidates)
                input_sent = self.template["source"].format(
                    " , ".join(["item_" + item_idx for item_idx in purchase_history]),
                    user_idx,
                    "item_" + negative_item,
                )
                output_sent = self.template["target"].format("no")

        return input_sent, output_sent


class evaluation_direct_yesno_dataset(Dataset):
    def __init__(self, args, users, all_items, all_history, task_group, mode):
        super().__init__()
        self.args = args
        self.evaluation_template_id = self.args.evaluation_template_id
        self.users = users
        self.all_items = all_items
        self.all_history = all_history
        self.template = task_group[self.evaluation_template_id]
        self.mode = mode

    def __len__(self):
        number = len(self.users) * (100 + 1)
        return number

    def __getitem__(self, index):
        polarity = "pos" if index % (1 + 100) == 0 else "neg"
        index = index // (1 + 100)
        user_idx = self.users[index]
        sequence = self.all_history[index]

        if self.template["input_first"] == "user":
            if polarity == "pos":
                if self.mode == "test":
                    target_item = sequence[-1]
                else:
                    target_item = sequence[-2]
                input_sent = self.template["source"].format(user_idx, target_item)
                output_sent = self.template["target"].format("yes")
            else:
                negative_items = self.all_items.copy()
                negative_items = list(set(negative_items) - set(sequence))
                negative_item = random.choice(negative_items)
                input_sent = self.template["source"].format(user_idx, negative_item)
                output_sent = self.template["target"].format("no")
        else:
            if polarity == "pos":
                if self.mode == "test":
                    target_item = sequence[-1]
                else:
                    target_item = sequence[-2]
                input_sent = self.template["source"].format(target_item, user_idx)
                output_sent = self.template["target"].format("yes")
            else:
                negative_items = self.all_items.copy()
                negative_items = list(set(negative_items) - set(sequence))
                negative_item = random.choice(negative_items)
                input_sent = self.template["source"].format(negative_item, user_idx)
                output_sent = self.template["target"].format("no")

        return input_sent, output_sent


class evaluation_direct_candidate_dataset(Dataset):
    def __init__(self, args, users, all_items, all_history, task_group, mode):
        super().__init__()
        self.args = args
        self.evaluation_template_id = self.args.evaluation_template_id
        self.users = users
        self.all_items = all_items
        self.all_history = all_history
        self.template = task_group[self.evaluation_template_id]
        self.mode = mode

    def __len__(self):
        length = len(self.users)
        return length

    def __getitem__(self, index):
        user_idx = self.users[index]
        sequence = self.all_history[index]

        if self.mode == "test":
            target_item = sequence[-1]
        else:
            target_item = sequence[-2]
        negative_items = self.all_items.copy()
        negative_items = list(set(negative_items) - set(sequence))
        negative_items = random.sample(negative_items, k=100)
        candidates = [target_item] + negative_items
        random.shuffle(candidates)

        if self.template["input_first"] == "user":
            input_sent = self.template["source"].format(
                user_idx, "items " + " , ".join(candidates)
            )
            output_sent = self.template["target"].format("item_" + target_item)
        else:
            input_sent = self.template["source"].format(
                "items " + " , ".join(candidates), user_idx
            )
            output_sent = self.template["target"].format("item_" + target_item)

        return input_sent, output_sent


def load_eval_dataloaders(
    args, tokenizer, method, mode, users, all_items, val_sequence, test_sequence,
):
    # load data
    # users, all_items, _, val_sequence, test_sequence = load_data(args, tokenizer)

    if mode == "validation":
        eval_sequence = val_sequence
    else:
        eval_sequence = test_sequence

    collator = Collator(tokenizer)

    if method == "sequential_item":
        # create sequential item dataset
        eval_sequential_item_dataset = evaluation_sequential_item_dataset(
            args, users, eval_sequence, task_subgroup_1, mode
        )

        # create sampler
        if args.distributed:
            sampler = DistributedSampler(eval_sequential_item_dataset)
        else:
            sampler = None
        dataloader = DataLoader(
            eval_sequential_item_dataset,
            batch_size=48,
            collate_fn=collator,
            shuffle=(sampler is None),
            sampler=sampler,
        )
    elif method == "sequential_yesno":
        # create sequential item dataset
        eval_sequential_yesno_dataset = evaluation_sequential_yesno_dataset(
            args, users, all_items, eval_sequence, task_subgroup_2
        )
        # create sampler
        if args.distributed:
            sampler = DistributedSampler(eval_sequential_yesno_dataset)
        else:
            sampler = None
        dataloader = DataLoader(
            eval_sequential_yesno_dataset,
            batch_size=48,
            collate_fn=collator,
            shuffle=(sampler is None),
            sampler=sampler,
        )
    elif method == "direct_yesno":
        # create sequential item dataset
        eval_direct_yesno_dataset = evaluation_direct_yesno_dataset(
            args, users, all_items, test_sequence, task_subgroup_3, mode
        )
        # create sampler
        if args.distributed:
            sampler = DistributedSampler(eval_direct_yesno_dataset)
        else:
            sampler = None
        dataloader = DataLoader(
            eval_direct_yesno_dataset,
            batch_size=48,
            collate_fn=collator,
            shuffle=(sampler is None),
            sampler=sampler,
        )
    else:
        assert method == "direct_candidate"
        eval_direct_candidate_dataset = evaluation_direct_candidate_dataset(
            args, users, all_items, test_sequence, task_subgroup_4, mode
        )
        # create sampler
        if args.distributed:
            sampler = DistributedSampler(eval_direct_candidate_dataset)
        else:
            sampler = None
        dataloader = DataLoader(
            eval_direct_candidate_dataset,
            batch_size=48,
            collate_fn=collator,
            shuffle=(sampler is None),
            sampler=sampler,
        )

    return dataloader


#################### meta data ####################
#################### title information data ####################


class title_dataset(Dataset):
    def __init__(
        self,
        args,
        all_items,
        task_subgroup_6,
        remap_all_items,
        meta_data,
        meta_dict,
        id2item,
    ):
        super().__init__()
        self.meta_data = meta_data
        self.meta_dict = meta_dict
        self.id2item = id2item
        self.all_items = all_items
        self.templates = task_subgroup_6
        self.num_templates = len(self.templates)
        self.remapped_all_items = remap_all_items

    def __len__(self):
        return len(self.all_items) * self.num_templates

    def __getitem__(self, index):
        item_index = index // self.num_templates
        item = str(self.all_items[item_index])
        meta_information = find_metadata(
            item, self.meta_data, self.meta_dict, self.id2item
        )
        title = meta_information["title"]

        remapped_item = self.remapped_all_items[item_index]
        template_index = index % self.num_templates
        template = self.templates[template_index]
        inputs = template["source"].format(title)
        output = template["target"].format(remapped_item)

        return inputs, output


class description_identification_dataset(Dataset):
    def __init__(
        self,
        args,
        all_items,
        task_subgroup_9,
        remapped_all_items,
        meta_data,
        meta_dict,
        id2item,
    ):
        super().__init__()
        self.meta_data = meta_data
        self.meta_dict = meta_dict
        self.id2item = id2item
        self.all_items = all_items
        self.templates = task_subgroup_9
        self.num_templates = len(self.templates)
        self.remapped_all_items = remapped_all_items

    def __len__(self):
        return len(self.all_items) * self.num_templates

    def __getitem__(self, index):
        item_index = index // self.num_templates
        item = str(self.all_items[item_index])
        meta_information = find_metadata(
            item, self.meta_data, self.meta_dict, self.id2item
        )
        if "description" in list(meta_information.keys()):
            description = meta_information["description"]
            description = (
                description.replace("\n-", "")
                .replace("\n", "")
                .replace("\t", " ")
                .replace("*", " ")
                .replace("  ", "")
            )
            if description.strip() == "":
                description = " , ".join(meta_information["categories"][0][1:])
        else:
            description = " , ".join(meta_information["categories"][0][1:])

        remapped_item = self.remapped_all_items[item_index]
        template_index = index % self.num_templates
        template = self.templates[template_index]
        inputs = template["source"].format(description)
        output = template["target"].format("item_" + remapped_item)

        return inputs, output


class category_dataset(Dataset):
    def __init__(
        self,
        args,
        all_items,
        task_subgroup_8,
        remapped_all_items,
        meta_data,
        meta_dict,
        id2item,
    ):
        super().__init__()
        self.meta_data = meta_data
        self.meta_dict = meta_dict
        self.id2item = id2item
        self.all_items = all_items
        self.templates = task_subgroup_8
        self.num_templates = len(self.templates)
        self.remapped_all_items = remapped_all_items

    def __len__(self):
        return len(self.all_items) * self.num_templates

    def __getitem__(self, index):
        item_index = index // self.num_templates
        item = str(self.all_items[item_index])
        meta_information = find_metadata(
            item, self.meta_data, self.meta_dict, self.id2item
        )
        categories = " , ".join(meta_information["categories"][0][1:])

        remapped_item = self.remapped_all_items[item_index]
        template_index = index % self.num_templates
        template = self.templates[template_index]
        inputs = template["source"].format(remapped_item)
        output = template["target"].format(categories)

        return inputs, output


def meta_loader(args, tokenizer, all_items, remapped_all_items):
    has_title_items = []
    has_title_remapped_items = []
    has_category_items = []
    has_category_remapped_items = []

    meta_data, meta_dict, id2item = load_meta(args)
    for item, remapped_item in zip(all_items, remapped_all_items):
        meta_information = find_metadata(item, meta_data, meta_dict, id2item)
        if "title" in list(meta_information.keys()):
            has_title_items.append(item)
            has_title_remapped_items.append(remapped_item)
        if "categories" in list(meta_information.keys()):
            has_category_items.append(item)
            has_category_remapped_items.append(remapped_item)

    pretrain_title_dataset = title_dataset(
        args,
        has_title_items,
        task_subgroup_6,
        has_title_remapped_items,
        meta_data,
        meta_dict,
        id2item,
    )
    pretrain_description_identification_dataset = description_identification_dataset(
        args,
        all_items,
        task_subgroup_9,
        remapped_all_items,
        meta_data,
        meta_dict,
        id2item,
    )
    pretrain_category_dataset = category_dataset(
        args,
        has_category_items,
        task_subgroup_8,
        has_category_remapped_items,
        meta_data,
        meta_dict,
        id2item,
    )

    collator = Collator(tokenizer)

    # create sampler
    if args.distributed:
        sampler = DistributedSampler(pretrain_title_dataset)
    else:
        sampler = None
    pretrain_title_dataloader = DataLoader(
        pretrain_title_dataset,
        batch_size=args.pretrain_title_batch,
        collate_fn=collator,
        shuffle=(sampler is None),
        sampler=sampler,
    )

    # create sampler
    if args.distributed:
        sampler = DistributedSampler(pretrain_description_identification_dataset)
    else:
        sampler = None
    pretrain_description_identification_dataloader = DataLoader(
        pretrain_description_identification_dataset,
        batch_size=args.pretrain_description_batch,
        collate_fn=collator,
        shuffle=(sampler is None),
        sampler=sampler,
    )

    # create sampler
    if args.distributed:
        sampler = DistributedSampler(pretrain_category_dataset)
    else:
        sampler = None
    pretrain_category_datalaoder = DataLoader(
        pretrain_category_dataset,
        batch_size=args.pretrain_category_batch,
        collate_fn=collator,
        shuffle=(sampler is None),
        sampler=sampler,
    )

    return (
        pretrain_title_dataloader,
        pretrain_description_identification_dataloader,
        pretrain_category_datalaoder,
    )

