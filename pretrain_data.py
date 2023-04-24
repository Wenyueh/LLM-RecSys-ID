import random
import argparse
from torch.utils.data import Dataset, DataLoader
from prompt import (
    task_subgroup_6,
    task_subgroup_8,
    task_subgroup_9,
    task_subgroup_10,
    task_subgroup_11,
)
import json
import gzip
import torch
from torch.utils.data.distributed import DistributedSampler
from data import load_data
import argparse
from transformers import AutoTokenizer


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
        output = template["target"].format("item_" + remapped_item)

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
        inputs = template["source"].format(categories)
        output = template["target"].format("item_" + remapped_item)

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
        batch_size=args.meta_title_batch,
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
        batch_size=args.meta_description_batch,
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
        batch_size=args.meta_category_batch,
        collate_fn=collator,
        shuffle=(sampler is None),
        sampler=sampler,
    )

    return (
        pretrain_title_dataloader,
        pretrain_description_identification_dataloader,
        pretrain_category_datalaoder,
    )


#################### review data #################
#################### ask item ####################


class review_item_dataset(Dataset):
    def __init__(
        self, args, reviews, all_items, task_subgroup_10, remapped_all_items,
    ):
        super().__init__()
        self.args = args
        self.reviews = reviews
        self.all_items = all_items
        self.templates = task_subgroup_10
        self.num_templates = len(self.templates)
        self.remapped_all_items = remapped_all_items

    def __len__(self):
        return len(self.reviews) * self.num_templates

    def __getitem__(self, index):
        review_index = index // self.num_templates
        review = self.reviews[review_index]

        text = review["text"]
        user_id = review["user"]
        item_id = review["item"]
        remapped_item_id = self.remapped_all_items[self.all_items.index(item_id)]

        template_index = index % self.num_templates
        template = self.templates[template_index]
        if template["input_first"] == "user":
            inputs = template["source"].format(user_id, text)
            output = template["target"].format("item_" + remapped_item_id)
        else:
            inputs = template["source"].format(text, user_id)
            output = template["target"].format("item_" + remapped_item_id)

        return inputs, output


class review_user_dataset(Dataset):
    def __init__(
        self, args, reviews, all_items, task_subgroup_11, remapped_all_items,
    ):
        super().__init__()
        self.args = args
        self.reviews = reviews
        self.all_items = all_items
        self.templates = task_subgroup_11
        self.num_templates = len(self.templates)
        self.remapped_all_items = remapped_all_items

    def __len__(self):
        return len(self.reviews) * self.num_templates

    def __getitem__(self, index):
        review_index = index // self.num_templates
        review = self.reviews[review_index]

        text = review["text"]
        user_id = review["user"]
        item_id = review["item"]
        remapped_item_id = self.remapped_all_items[self.all_items.index(item_id)]

        template_index = index % self.num_templates
        template = self.templates[template_index]
        if template["input_first"] == "item":
            # don't need to concatenate "item_" here, it's in the prompt
            inputs = template["source"].format(remapped_item_id, text)
            output = template["target"].format("User_" + user_id)
        else:
            inputs = template["source"].format(text, remapped_item_id)
            output = template["target"].format("User_" + user_id)

        return inputs, output


def load_reviews(args):
    with open(args.data_dir + args.task + "/train_reviews.json", "r") as f:
        train_reviews = json.load(f)

    return train_reviews


def review_loader(args, tokenizer, all_items, remapped_all_items):
    # load review data
    reviews = load_reviews(args)

    # create dataset
    pretrain_review_item_dataset = review_item_dataset(
        args, reviews, all_items, task_subgroup_10, remapped_all_items,
    )
    pretrain_review_user_dataset = review_user_dataset(
        args, reviews, all_items, task_subgroup_11, remapped_all_items,
    )
    collator = Collator(tokenizer)

    if args.distributed:
        sampler = DistributedSampler(pretrain_review_item_dataset)
    else:
        sampler = None
    pretrain_review_item_dataloader = DataLoader(
        pretrain_review_item_dataset,
        batch_size=args.review_item_batch,
        collate_fn=collator,
        shuffle=(sampler is None),
        sampler=sampler,
    )

    if args.distributed:
        sampler = DistributedSampler(pretrain_review_user_dataset)
    else:
        sampler = None
    pretrain_review_user_dataloader = DataLoader(
        pretrain_review_user_dataset,
        batch_size=args.review_user_batch,
        collate_fn=collator,
        shuffle=(sampler is None),
        sampler=sampler,
    )

    return (
        pretrain_review_item_dataloader,
        pretrain_review_user_dataloader,
    )


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

    parser.add_argument("--item_representation", type=str, default=None)
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

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    new_tokens = []
    for x in range(12101):
        new_token = "<extra_id_{}>".format(x)
        new_tokens.append(new_token)
    (
        users,
        all_items,
        train_sequence,
        val_sequence,
        test_sequence,
        remapped_all_items,
    ) = load_data(args, tokenizer)

    reviews = load_reviews(args)

    review_dataset = review_item_dataset(
        args, reviews, all_items, task_subgroup_11, remapped_all_items,
    )

    review_dataset = review_item_dataset(
        args, reviews, all_items, task_subgroup_10, remapped_all_items,
    )

    print(review_dataset[0])
    print(review_dataset[1])
    print(review_dataset[2])
    print(review_dataset[3])
    print(review_dataset[4])
