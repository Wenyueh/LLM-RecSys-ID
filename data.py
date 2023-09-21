from sklearn.cluster import SpectralClustering
import random
import argparse
from torch.utils.data import Dataset, DataLoader
from prompt import (
    task_subgroup_1,
    task_subgroup_2,
    task_subgroup_3,
    task_subgroup_4,
    task_subgroup_5,
)
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import math
from torch.utils.data.distributed import DistributedSampler
from item_rep_method import (
    random_number,
    random_number_remove_zero,
    no_tokenization,
    #item_resolution,
    change_base,
    build_category_map,
    content_based_representation,
    load_meta,
    title_representation,
    CF_representation,
    CF_representation_optimal_width,
    create_CF_embedding,
    create_CF_embedding_optimal_width,
    load_hybrid,
    hybrid_representation,
    create_hybrid_embedding,
    build_category_map_modified_yelp,
    content_based_representation_modified_yelp,
)
import time
import numpy as np
from utils import create_category_embedding
from CF_index import (
    construct_indices_from_cluster,
    construct_indices_from_cluster_optimal_width,
)


def load_data(args, tokenizer):
    # set_seed(args)
    if args.data_order == "remapped_sequential":
        if args.remapped_data_order == "original":
            data_dir = args.data_dir + args.task + "/remapped_sequential_data.txt"
            with open(data_dir, "r") as f:
                data = f.read()
            data = data.split("\n")[:-1]
        elif args.remapped_data_order == "short_to_long":
            data_dir = (
                args.data_dir
                + args.task
                + "/short_to_long_remapped_sequential_data.txt"
            )
            with open(data_dir, "r") as f:
                data = f.read()
            data = data.split("\n")[:-1]
        elif args.remapped_data_order == "long_to_short":
            data_dir = (
                args.data_dir
                + args.task
                + "/long_to_short_remapped_sequential_data.txt"
            )
            with open(data_dir, "r") as f:
                data = f.read()
            data = data.split("\n")[:-1]
        else:
            assert args.remapped_data_order == "randomize"
            data_dir = (
                args.data_dir + args.task + "/randomize_remapped_sequential_data.txt"
            )
            with open(data_dir, "r") as f:
                data = f.read()
            data = data.split("\n")[:-1]
    else:
        assert args.data_order == "random"
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
        remap_fct = random_number(
            min_number=args.min_random_number, max_number=args.max_random_number
        )

    elif args.item_representation == "no_tokenization":
        remap_fct = no_tokenization(args)

    elif args.item_representation == "content_based":
        meta_data, meta_dict, id2item = load_meta(args)
        category_dict, level_categories = build_category_map(
            args, meta_data, meta_dict, id2item
        )
        print("--- level_categories ---")
        print(level_categories[0])
        remap_fct = lambda index: content_based_representation(
            args, index, category_dict, level_categories
        )
        # category_dict = build_category_map_modified_yelp(args)
        # remap_fct = lambda index: content_based_representation_modified_yelp(
        #    index, category_dict
        # )

    elif args.item_representation == "item_resolution":
        if args.data_order == "random":
            if args.resolution == 1:
                random_fct = random_number(
                    min_number=args.min_random_number, max_number=args.max_random_number
                )
            else:
                random_fct = random_number_remove_zero(
                    min_number=args.min_random_number, max_number=args.max_random_number
                )
            remap_fct = lambda x: item_resolution(
                resolution=args.resolution, overlap=args.overlap
            )(change_base(random_fct(x), args.base))
        else:
            remap_fct = lambda x: item_resolution(
                resolution=args.resolution, overlap=args.overlap
            )(change_base(x, args.base))

    elif args.item_representation == "title":
        meta_data, meta_dict, id2item = load_meta(args)
        remap_fct = lambda x: title_representation(x, meta_data, meta_dict, id2item)

    elif args.item_representation == "CF":
        assert args.data_order == "remapped_sequential"
        if not args.optimal_width_in_CF:
            print("---do not use optimal width in CF---")
            mapping, _ = construct_indices_from_cluster(args)
            remap_fct = lambda x: CF_representation(x, mapping)
        else:
            print("---do use optimal width in CF---")
            mapping, _ = construct_indices_from_cluster_optimal_width(args)
            remap_fct = lambda x: CF_representation_optimal_width(x, mapping)
        print("---finish loading CF mapping---")

    elif args.item_representation == "hybrid":
        assert args.data_order == "random"
        content_based_representation_dict, _ = load_hybrid(args)
        remap_fct = lambda x: hybrid_representation(
            x, content_based_representation_dict
        )
        print("---finish loading hybrid mapping---")

    for one_user in tqdm(data):
        splittion_point = one_user.index(" ")
        user = one_user[:splittion_point]
        items = one_user[splittion_point + 1 :].split(" ")
        if args.item_representation != "None":
            remapped_items = [remap_fct(item) for item in items]
        else:
            # remapped_items = items
            remapped_items = [str(int(item) + 1000) for item in items]

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


def calculate_whole_word_ids_remove_number_begin(tokenized_text, input_ids):
    whole_word_ids = []
    curr = 0
    for i in range(len(tokenized_text)):
        if tokenized_text[i] == "<pad>":
            curr = 0
        if tokenized_text[i] == "▁":
            if tokenized_text[i + 1][0] in [
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
            ]:
                whole_word_ids.append(curr)
            else:
                curr += 1
                whole_word_ids.append(curr)
        elif tokenized_text[i].startswith("▁") and tokenized_text[i][1] in [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        ]:
            whole_word_ids.append(curr)
        elif tokenized_text[i].startswith("▁"):
            curr += 1
            whole_word_ids.append(curr)
        else:
            whole_word_ids.append(curr)
    return whole_word_ids[: len(input_ids) - 1] + [0]  # [0] for </s>


def calculate_whole_word_ids_title(tokenized_text, input_ids):
    whole_word_ids = []
    curr = 0
    in_title = False
    for i in range(len(tokenized_text)):
        if "[" in tokenized_text[i]:
            in_title = True
        elif "]" in tokenized_text[i]:
            in_title = False
        if tokenized_text[i] == "<pad>":
            curr = 0
        if not in_title:
            if tokenized_text[i].startswith("▁"):
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        else:
            whole_word_ids.append(curr)
    return whole_word_ids[: len(input_ids) - 1] + [0]  # [0] for </s>


def position_embedding(args, tokenized_text, input_ids):
    if args.item_representation != "title":
        whole_word_ids = calculate_whole_word_ids(tokenized_text, input_ids)
    else:
        whole_word_ids = calculate_whole_word_ids_title(tokenized_text, input_ids)

    # item position embedding
    item_starts = [
        i
        for i, c in enumerate(tokenized_text)
        if "item" in c and "_" in tokenized_text[i + 1]
    ]
    item_whole_word_starts = [whole_word_ids[start] for start in item_starts]
    in_item_indices = [
        [
            i
            for i, whole_word_id in enumerate(whole_word_ids)
            if whole_word_id == item_whole_word_start
        ]
        for item_whole_word_start in item_whole_word_starts
    ]
    # user position embedding
    user_starts = [
        i
        for i, c in enumerate(tokenized_text)
        if "user" in c and "_" in tokenized_text[i + 1]
    ]
    user_whole_word_starts = [whole_word_ids[start] for start in user_starts]
    in_user_indices = [
        [
            i
            for i, whole_word_id in enumerate(whole_word_ids)
            if whole_word_id == user_whole_word_start
        ]
        for user_whole_word_start in user_whole_word_starts
    ]

    # compute mapping for item and user
    position_map = {}
    for i in range(len(in_item_indices)):
        one_item_indices = in_item_indices[i]
        for item_position, sentence_index in enumerate(one_item_indices):
            position_map[sentence_index] = item_position + 1
    for i in range(len(in_user_indices)):
        one_user_indices = in_user_indices[i]
        for user_position, sentence_index in enumerate(one_user_indices):
            position_map[sentence_index] = user_position + 100

    # compose final position embedding for the whole sentence
    sentence_position_index = []
    for i in range(len(tokenized_text)):
        if i not in position_map:
            sentence_position_index.append(0)
        else:
            sentence_position_index.append(position_map[i])
    return sentence_position_index


class Collator:
    def __init__(self, args, tokenizer):
        self.args = args
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
            if (
                self.args.whole_word_embedding == "shijie"
                or self.args.whole_word_embedding == "None"
            ):
                if self.args.item_representation != "title":
                    whole_word_id = calculate_whole_word_ids(tokenized_text, input_id)
                    # whole_word_id = calculate_whole_word_ids_remove_number_begin(
                    #    tokenized_text, input_id
                    # )
                else:
                    whole_word_id = calculate_whole_word_ids_title(
                        tokenized_text, input_id
                    )
            else:
                assert self.args.whole_word_embedding == "position_embedding"
                whole_word_id = position_embedding(self.args, tokenized_text, input_id)
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
    collator = Collator(args, tokenizer)

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

    collator = Collator(args, tokenizer)

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
            batch_size=24,
            collate_fn=collator,
            shuffle=False,  # (sampler is None),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--task", type=str, default="beauty")

    parser.add_argument("--max_history", type=int, default=20)
    parser.add_argument("--sequential_num", type=int, default=5)
    parser.add_argument("--negative_sample", type=int, default=2)
    parser.add_argument("--yes_no_sample", type=int, default=5)
    parser.add_argument("--direct_item_proportion", type=int, default=2)

    parser.add_argument("--train_sequential_item_batch", type=int, default=4)
    parser.add_argument("--train_sequential_yesno_batch", type=int, default=4)
    parser.add_argument("--train_direct_yesno_batch", type=int, default=4)
    parser.add_argument("--train_direct_candidate_batch", type=int, default=4)
    parser.add_argument("--train_direct_straightforward_batch", type=int, default=4)

    parser.add_argument("--evaluation_template_id", type=int, default=0)

    parser.add_argument("--data_order", type=str, default="random")
    parser.add_argument("--item_representation", type=str, default=None, help="CF")
    parser.add_argument(
        "--whole_word_embedding", type=str, default="position_embedding"
    )
    parser.add_argument(
        "--max_random_number",
        type=int,
        default=30000,
        help="must be larger than number of items/users, this is the range of item id random mapping",
    )
    parser.add_argument(
        "--min_random_number",
        type=int,
        default=1000,
        help="this is the lower bound of the range of item id random mapping",
    )
    parser.add_argument(
        "--base",
        type=int,
        default=10,
        help="change the base for number representation",
    )
    parser.add_argument(
        "--user_representation", type=str, default=None, help="no_tokenization"
    )
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--remove_last_item", action="store_true")
    parser.add_argument("--remove_first_item", action="store_true")

    parser.add_argument("--distributed", action="store_true")

    parser.add_argument("--cluster_size", type=int, default=100)
    parser.add_argument("--cluster_number", type=int, default=55)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    new_tokens = []
    for x in range(12101):
        new_token = "<extra_id_{}>".format(x)
        new_tokens.append(new_token)
    # tokenizer = create_category_embedding(tokenizer)

    # tokenizer = create_CF_embedding(args, tokenizer)

    content_based_representation_dict, vocabulary = load_hybrid(args)
    tokenizer = create_hybrid_embedding(vocabulary, tokenizer)

    (
        users,
        all_items,
        train_sequence,
        val_sequence,
        test_sequence,
        remapped_all_items,
    ) = load_data(args, tokenizer)

    ## sequential dataset

    """
    train_dataset = sequential_item_dataset(
        args, users, train_sequence, task_subgroup_1
    )
    val_dataset = evaluation_sequential_item_dataset(
        args, users, val_sequence, task_subgroup_1, "validation"
    )
    test_dataset = evaluation_sequential_item_dataset(
        args, users, test_sequence, task_subgroup_1, "test"
    )
    # for d in tqdm(test_dataset):
    #    print(d)
    #    time.sleep(5)
    for j in range(1, 1000):
        print("user_{} ".format(j))
        print("train data")
        for data in tqdm(train_dataset):
            if "user_{} ".format(j) in data[0] or "User_{} ".format(j) in data[0]:
                print(data)
                time.sleep(5)
                break
        print("val data")
        for vald in tqdm(val_dataset):
            if "user_{} ".format(j) in vald[0] or "User_{} ".format(j) in vald[0]:
                print(vald)
                time.sleep(5)
                break
        print("test data")
        for testd in tqdm(test_dataset):
            if "user_{} ".format(j) in testd[0] or "User_{} ".format(j) in testd[0]:
                print(testd)
                time.sleep(5)
                break
        print("***")
        print("***")
        print("***")

    val_dataset = evaluation_sequential_item_dataset(
        args, users, val_sequence, task_subgroup_1, "validation"
    )
    for i in range(len(val_dataset)):
        for j in range(len(train_dataset)):
            if (
                "user_{}".format(i) in train_dataset[j][0]
                or "User_{}".format(i) in train_dataset[j][0]
            ):
                print(train_dataset[j])
                break
        print(val_dataset[i])
        print("***")
        print("***")
        time.sleep(5)
    """

    # dataset = direct_candidate_dataset(
    #    args, users, all_items, test_sequence, task_subgroup_4
    # )

    # dataset = direct_yesno_dataset(
    #    args, users, all_items, test_sequence, task_subgroup_3
    # )

    # print(len(dataset))
    # for i in range(100):
    #    print(dataset[i])
    # print(dataset[0][0])
    # for i in range(len(dataset)):
    #    if "user_1 " in dataset[i][0] or "User_1 " in dataset[i][0]:
    #        print(dataset[i])
    # for i in tqdm(range(len(dataset))):
    #    a = dataset[0]

    ## validation dataset
    # dataset = evaluation_sequential_item_dataset(
    #    args, users, val_sequence, task_subgroup_1
    # )
    # dataset = evaluation_sequential_yesno_dataset(
    #    args, users, all_items, val_sequence, task_subgroup_2
    # )
    # dataset = evaluation_direct_yesno_dataset(
    #    args, users, all_items, test_sequence, task_subgroup_3, "val"
    # )
    # dataset = evaluation_direct_candidate_dataset(
    #    args, users, all_items, test_sequence, task_subgroup_4, "test"
    # )

    ## test dataset
    # dataset = evaluation_sequential_item_dataset(
    #    args, users, test_sequence, task_subgroup_1
    # )
    # dataset = evaluation_sequential_yesno_dataset(
    #    args, users, all_items, test_sequence, task_subgroup_2
    # )
    # dataset = evaluation_direct_yesno_dataset(
    #    args, users, all_items, test_sequence, task_subgroup_3, "test"
    # )
    # dataset = evaluation_direct_candidate_dataset(
    #    args, users, all_items, test_sequence, task_subgroup_4, "test"
    # )

    # print(len(dataset))
    # print(dataset[0])
    # print(dataset[len(dataset) - 1])
    # for i in tqdm(range(len(dataset))):
    #    a = dataset[0]

    (
        train_sequential_item_dataloader,
        train_sequential_yesno_dataloader,
        train_direct_yesno_dataloader,
        train_direct_candidate_dataloader,
        train_direct_straightforward_dataloader,
        task_data_lengths,
        all_items,
    ) = load_train_dataloaders(
        args, tokenizer, users, all_items, train_sequence, test_sequence
    )

    for batch in train_sequential_item_dataloader:
        input_ids = batch[0]
        for one in input_ids:
            print(tokenizer.convert_ids_to_tokens(one))
            # print(tokenizer.decode(input_ids[1]))
            # print(tokenizer.decode(input_ids[2]))
            # print(tokenizer.decode(input_ids[3]))
            print("***")
            time.sleep(5)

    """
    iterator = iter(train_sequential_item_dataloader)
    print(next(iterator))
    print("***")
    print(next(iterator))
    print("***")
    print(next(iterator))
    print("***")
    print(next(iterator))
    """
