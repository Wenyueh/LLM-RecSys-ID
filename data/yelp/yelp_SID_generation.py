import pickle as pkl
import json
import random
from collections import Counter
import numpy as np
from tqdm import tqdm


# given an item index, return its category information
def find_metadata(index, meta_data, meta_dict, id2item):
    asin = id2item[index]  # index is string type
    meta_data_position = meta_dict[asin]
    index_meta_data = meta_data[meta_data_position]
    return index_meta_data


# retain the categories that occur more than 10 times in the dataset, remove long tail categories
def filter_categories(meta_data, meta_dict, id2item):
    indices = list(range(1, 20034))
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


def build_category_map(meta_data, meta_dict, id2item):
    category_dict = {}
    category_counts = {}
    # randomize it to avoid data leakage
    # beauty 12102
    # toys 11925
    # sports 18358
    # yelp 20034
    indices = list(range(1, 20034))
    random.shuffle(indices)

    # to count in yelp time
    all_possible_categories = {}

    for i in indices:
        index = str(i)
        categories = find_metadata(index, meta_data, meta_dict, id2item)["categories"]
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

    filtered_categories = filter_categories(meta_data, meta_dict, id2item)
    for i in indices:
        index = str(i)
        # find categories in meta data
        categories = find_metadata(index, meta_data, meta_dict, id2item)["categories"]
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
        category_dict[index] = categories + [str(category_counts[tuple(categories)])]

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


if __name__ == "__main__":
    id2item_dir = "datamaps.json"
    meta_dir = "meta_data.pkl"

    meta_data = []
    with open(meta_dir, "rb") as f:
        meta_data = pkl.load(f)
    meta_dict = {}
    for i, meta_item in enumerate(meta_data):
        meta_dict[meta_item["business_id"]] = i

    with open(id2item_dir, "r") as f:
        datamaps = json.load(f)

    id2item = datamaps["id2item"]

    all_possible_categories = filter_categories(meta_data, meta_dict, id2item)

    print("print total number of filtered categories: ")
    print(len(all_possible_categories))

    (category_dict, level_categories) = build_category_map(
        meta_data, meta_dict, id2item
    )

    # for each category, count whether it occurs in level 1 the most, if yes, delete them all in other levels, recursively
    left_categories_inorder = [v[:-1] for k, v in category_dict.items()]
    left_categories = list(set([a for k, v in category_dict.items() for a in v[:-1]]))

    print("level 1 categories are:")
    print(left_categories_inorder[0])

    level_one = [cs[0] for cs in left_categories_inorder if len(cs) > 0]
    level_two = [cs[1] for cs in left_categories_inorder if len(cs) > 1]
    level_three = [cs[2] for cs in left_categories_inorder if len(cs) > 2]
    level_four = [cs[3] for cs in left_categories_inorder if len(cs) > 3]
    level_five = [cs[4] for cs in left_categories_inorder if len(cs) > 4]

    #### compute level one categories
    filtered_level_one = []
    for c in left_categories:
        one_number = level_one.count(c)
        two_number = level_two.count(c)
        three_number = level_three.count(c)
        four_number = level_four.count(c)
        five_number = level_five.count(c)
        if (
            np.argmax([one_number, two_number, three_number, four_number, five_number])
            == 0
        ):
            filtered_level_one.append(c)

    modified_category_dict = {}
    for k, v in tqdm(category_dict.items()):
        if list(set(filtered_level_one).intersection(v)) != []:
            select_one = list(set(filtered_level_one).intersection(v))
            rest = []
            for one_cat in v:
                if one_cat not in select_one:
                    rest.append(one_cat)
            modified_category_dict[k] = [select_one[0]] + rest
        else:
            select_one = {}
            for candidate in filtered_level_one:
                select_one[candidate] = len(
                    [
                        l
                        for l in left_categories_inorder
                        if set([candidate, v[1]]).issubset(set(l))
                    ]
                )
            select_one = sorted(select_one.items(), key=lambda x: x[1], reverse=True)
            c = select_one[0][0]
            modified_category_dict[k] = [c] + v

    #### compute level two categories
    filtered_level_two = []
    level_two_left_categories_inorder = [
        v[:-1] for k, v in modified_category_dict.items()
    ]
    level_two_left_categories = list(
        set([a for k, v in modified_category_dict.items() for a in v[:-1]])
    )
    level_two_level_one = [
        cs[0] for cs in level_two_left_categories_inorder if len(cs) > 0
    ]
    level_two_level_two = [
        cs[1] for cs in level_two_left_categories_inorder if len(cs) > 1
    ]
    level_two_level_three = [
        cs[2] for cs in level_two_left_categories_inorder if len(cs) > 2
    ]
    level_two_level_four = [
        cs[3] for cs in level_two_left_categories_inorder if len(cs) > 3
    ]
    level_two_level_five = [
        cs[4] for cs in level_two_left_categories_inorder if len(cs) > 4
    ]
    for c in level_two_left_categories:
        one_number = level_two_level_one.count(c)
        two_number = level_two_level_two.count(c)
        three_number = level_two_level_three.count(c)
        four_number = level_two_level_four.count(c)
        five_number = level_two_level_five.count(c)
        if (
            np.argmax([one_number, two_number, three_number, four_number, five_number])
            == 1
        ):
            filtered_level_two.append(c)

    level_two_category_dict = {}
    for k, v in tqdm(modified_category_dict.items()):
        if list(set(filtered_level_two).intersection(v)) != []:
            select_one = list(set(filtered_level_two).intersection(v))
            rest = []
            for one_cat in v:
                if one_cat not in select_one:
                    rest.append(one_cat)
            level_two_category_dict[k] = [rest[0]] + [select_one[0]] + rest[1:]
        else:
            select_one = {}
            for candidate in filtered_level_two:
                if len(v) > 2:
                    select_one[candidate] = len(
                        [
                            l
                            for l in level_two_left_categories_inorder
                            if set([v[1], candidate, v[2]]).issubset(set(l))
                        ]
                    )
                else:
                    select_one[candidate] = len(
                        [
                            l
                            for l in level_two_left_categories_inorder
                            if set([candidate, v[1]]).issubset(set(l))
                        ]
                    )
            select_one_item = sorted(
                select_one.items(), key=lambda x: x[1], reverse=True
            )
            c = select_one_item[0][0]
            if select_one[c] != 0:
                c = select_one_item[0][0]
                level_two_category_dict[k] = [v[0]] + [c] + v[1:]
            else:
                assert v[1] not in filtered_level_one
                level_two_category_dict[k] = v

    #### compute level three categories
    filtered_level_three = []
    level_three_left_categories_inorder = [
        v[:-1] for k, v in level_two_category_dict.items()
    ]
    level_three_left_categories = list(
        set([a for k, v in level_two_category_dict.items() for a in v[:-1]])
    )
    level_three_level_one = [
        cs[0] for cs in level_three_left_categories_inorder if len(cs) > 0
    ]
    level_three_level_two = [
        cs[1] for cs in level_three_left_categories_inorder if len(cs) > 1
    ]
    level_three_level_three = [
        cs[2] for cs in level_three_left_categories_inorder if len(cs) > 2
    ]
    level_three_level_four = [
        cs[3] for cs in level_three_left_categories_inorder if len(cs) > 3
    ]
    level_three_level_five = [
        cs[4] for cs in level_three_left_categories_inorder if len(cs) > 4
    ]
    for c in level_three_left_categories:
        one_number = level_three_level_one.count(c)
        two_number = level_three_level_two.count(c)
        three_number = level_three_level_three.count(c)
        four_number = level_three_level_four.count(c)
        five_number = level_three_level_five.count(c)
        if (
            np.argmax([one_number, two_number, three_number, four_number, five_number])
            == 2
        ):
            filtered_level_three.append(c)

    level_three_category_dict = {}
    for k, v in tqdm(level_two_category_dict.items()):
        if list(set(filtered_level_three).intersection(v)) != []:
            select_one = list(set(filtered_level_three).intersection(v))
            rest = []
            for one_cat in v:
                if one_cat not in select_one:
                    rest.append(one_cat)
            level_three_category_dict[k] = rest[:-1][:2] + [select_one[0]] + rest[2:]
        else:
            select_one = {}
            for candidate in filtered_level_three:
                if len(v[:-1]) > 3:
                    select_one[candidate] = len(
                        [
                            l
                            for l in level_three_left_categories_inorder
                            if set([v[0], v[1], v[2], v[3], candidate]).issubset(set(l))
                        ]
                    )
                elif len(v[:-1]) > 2:
                    select_one[candidate] = len(
                        [
                            l
                            for l in level_three_left_categories_inorder
                            if set([v[0], v[1], v[2], candidate]).issubset(set(l))
                        ]
                    )
                elif len(v[:-1]) > 1:
                    select_one[candidate] = len(
                        [
                            l
                            for l in level_three_left_categories_inorder
                            if set([v[0], v[1], candidate]).issubset(set(l))
                        ]
                    )
            if select_one != {}:
                select_one_item = sorted(
                    select_one.items(), key=lambda x: x[1], reverse=True
                )
                c = select_one_item[0][0]
                if select_one[c] != 0:
                    level_three_category_dict[k] = v[:2] + [c] + v[2:]
                else:
                    assert (
                        v[2] not in filtered_level_two
                        and v[2] not in filtered_level_one
                    )
                    level_three_category_dict[k] = v
            else:
                level_three_category_dict[k] = v

    ##### save some intermediate results
    with open("level_three_category_dict.json", "w") as f:
        json.dump(level_three_category_dict, f)

    print("an example of item category with three levels:")
    print(level_three_category_dict["17"])

    reordered_dict = {}
    for k, v in level_three_category_dict.items():
        new_v = [v[i] for i in range(len(v)) if (not v[i].isdigit() or i == 0)][
            :4
        ]  # and v[i] != 'Diners'
        reordered_dict[k] = new_v

    category_counts = {}
    for k, v in reordered_dict.items():
        if tuple(v) in category_counts:
            category_counts[tuple(v)] += 1
        else:
            category_counts[tuple(v)] = 1

    single_category_counts = {}
    for k, v in reordered_dict.items():
        for c in v:
            if c in single_category_counts:
                single_category_counts[c] += 1
            else:
                single_category_counts[c] = 1

    all_categories = list(set([k for k, v in single_category_counts.items()]))
    print("print the number of categories")
    print(len(all_categories))

    filtered_dict = {}
    for k, v in reordered_dict.items():
        if single_category_counts[v[-1]] == 1:
            new_v = v[:-1]
            filtered_dict[k] = new_v
        else:
            filtered_dict[k] = v

    filtered_single_category_counts = {}
    for k, v in filtered_dict.items():
        for c in v:
            if c in filtered_single_category_counts:
                filtered_single_category_counts[c] += 1
            else:
                filtered_single_category_counts[c] = 1

    filtered_category_counts = {}
    for k, v in filtered_dict.items():
        if tuple(v) in filtered_category_counts:
            filtered_category_counts[tuple(v)] += 1
        else:
            filtered_category_counts[tuple(v)] = 1

    multi = [k for k, v in filtered_category_counts.items() if v > 1]

    closed_categories = {}
    for number, categories in filtered_dict.items():
        closed_categories[number] = categories
        if filtered_category_counts[tuple(categories)] == 1:
            overlap = {m: len(set(categories).intersection(set(m))) for m in multi}
            similar_categories = [
                a
                for a, v in overlap.items()
                if (v == len(categories) - 1 and len(a) == len(categories))
            ]
            if len(similar_categories) == 1:
                new_v = similar_categories[0]
                closed_categories[number] = new_v

    with open("closed_categories.json", "w") as f:
        json.dump(closed_categories, f)

    closed_category_counts = {}
    for k, v in closed_categories.items():
        if tuple(v) in closed_category_counts:
            closed_category_counts[tuple(v)] += 1
        else:
            closed_category_counts[tuple(v)] = 1
