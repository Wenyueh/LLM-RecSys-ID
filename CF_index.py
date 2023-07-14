from sklearn.cluster import SpectralClustering
import json
import time
import networkx as nx
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import networkx as nx
import argparse
import os

from numpy import linalg as LA
import scipy
from scipy.sparse import csgraph

# from sklearn.cluster import SpectralClustering


def eigenDecomposition(A):
    L = csgraph.laplacian(A, normed=False)
    n_components = A.shape[0]

    eigenvalues, eigenvectors = LA.eig(L)
    eigenvalues = sorted(eigenvalues, reverse=True)

    index_largest_gap = np.argmax(np.diff(eigenvalues))
    nb_clusters = index_largest_gap + 1

    return nb_clusters


def composite_function(f, g):
    return lambda x: f(g(x))


def one_further_indexing(
    which_group,
    group_labels,
    sorted_pair_freqs,
    number_of_clusters,
    item_CF_index_map,
    reverse_fcts,
    mode,
):
    # select items in this large cluster
    one_subcluster_items = [
        item for item, l in enumerate(group_labels) if l == which_group
    ]
    # edges within the subcluster
    subcluster_pairs = [
        sorted_pair_freq
        for sorted_pair_freq in sorted_pair_freqs
        if sorted_pair_freq[0][0] in one_subcluster_items
        and sorted_pair_freq[0][1] in one_subcluster_items
    ]
    # remap the item indices
    item_map = {
        old_item_index: i for i, old_item_index in enumerate(one_subcluster_items)
    }
    reverse_item_map = {
        i: old_item_index for i, old_item_index in enumerate(one_subcluster_items)
    }
    # modify the subcluster pairs by item_map
    remapped_subcluster_pairs = [
        (
            (item_map[subcluster_pair[0][0]], item_map[subcluster_pair[0][1]]),
            subcluster_pair[1],
        )
        for subcluster_pair in subcluster_pairs
    ]

    # create new matrix
    sub_matrix_size = len(item_map)
    sub_adjacency_matrix = np.zeros((sub_matrix_size, sub_matrix_size))
    for pair, freq in remapped_subcluster_pairs:
        sub_adjacency_matrix[pair[0], pair[1]] = freq
        sub_adjacency_matrix[pair[1], pair[0]] = freq

    # compute optimal number of subclusters
    n = eigenDecomposition(sub_adjacency_matrix)
    if n <= number_of_clusters:
        if n != 1:
            number_of_clusters = n
        else:
            number_of_clusters = 2

    # clustering
    sub_clustering = SpectralClustering(
        n_clusters=number_of_clusters,
        assign_labels="cluster_qr",
        random_state=0,
        affinity="precomputed",
    ).fit(sub_adjacency_matrix)
    sub_labels = sub_clustering.labels_.tolist()

    # remap the index to the actual item
    reversal = lambda x: x
    for reverse_fct in reverse_fcts:
        reversal = composite_function(
            reverse_fct, reversal
        )  # lambda x: reverse_fct(reversal(x))

    for i, label in enumerate(sub_labels):
        item_CF_index_map[reversal(reverse_item_map[i])].append(str(label))

    # concatenate the new reverse function
    new_reverse_fcts = [lambda y: reverse_item_map[y]] + reverse_fcts

    return sub_labels, remapped_subcluster_pairs, item_CF_index_map, new_reverse_fcts


def compute_item_CF_index_map(labels, sorted_pair_freqs, item_CF_index_map, N):
    level_one = labels

    reverse_fcts = [lambda x: x]
    for which_group in range(N):
        if level_one.count(which_group) > N:
            (
                a_labels,
                remapped_subcluster_pairs,
                item_CF_index_map,
                level_two_reverse_fcts,
            ) = one_further_indexing(
                which_group,
                labels,
                sorted_pair_freqs,
                N,
                item_CF_index_map,
                reverse_fcts,
                2,
            )
            for which_sub_group in range(N):
                if a_labels.count(which_sub_group) > N:
                    (_, _, item_CF_index_map, _,) = one_further_indexing(
                        which_sub_group,
                        a_labels,
                        remapped_subcluster_pairs,
                        N,
                        item_CF_index_map,
                        level_two_reverse_fcts,
                        3,
                    )

    return item_CF_index_map


def within_category_spectral_clustering(args, category_items, category_pairs):
    # remap items and corresponding pairs
    item_map = {old_item_index: i for i, old_item_index in enumerate(category_items)}
    reverse_item_map = {
        i: old_item_index for i, old_item_index in enumerate(category_items)
    }
    remapped_category_pairs = [
        (
            (item_map[category_pair[0][0]], item_map[category_pair[0][1]]),
            category_pair[1],
        )
        for category_pair in category_pairs
    ]
    matrix_size = len(item_map)
    adjacency_matrix = np.zeros((matrix_size, matrix_size))
    for pair, freq in remapped_category_pairs:
        adjacency_matrix[pair[0], pair[1]] = freq
        adjacency_matrix[pair[1], pair[0]] = freq

    # here adjacency matrix is the affinity matrix
    number_of_clusters = args.cluster_size
    clustering = SpectralClustering(
        n_clusters=number_of_clusters,
        assign_labels="cluster_qr",
        random_state=0,
        affinity="precomputed",
    ).fit(adjacency_matrix)

    labels = clustering.labels_.tolist()
    item_CF_index_map = {i: [str(label)] for i, label in enumerate(labels)}
    item_CF_index_map = compute_item_CF_index_map(
        labels, remapped_category_pairs, item_CF_index_map, number_of_clusters
    )
    # add back category information
    item_CF_index_map = {reverse_item_map[k]: v for k, v in item_CF_index_map.items()}
    return item_CF_index_map


############ pure CF ############
def compute_index(item_CF_index_map):
    reformed_item_CF_index_map = {}
    for item, labels in item_CF_index_map.items():
        reformed_item_CF_index_map[item] = [
            "-".join(labels[: i + 1]) for i in range(len(labels))
        ]

    vocabulary = []
    enumeration_by_group = {}
    full_index = {}
    for item, clusters in reformed_item_CF_index_map.items():
        if tuple(clusters) not in enumeration_by_group:
            v = clusters + ["A" + str(0)]
            full_index[item] = "".join(["<{}>".format(a) for a in v])
            vocabulary += ["<{}>".format(a) for a in v]
            enumeration_by_group[tuple(clusters)] = 1
        else:
            v = clusters + ["A" + str(enumeration_by_group[tuple(clusters)])]
            full_index[item] = "".join(["<{}>".format(a) for a in v])
            vocabulary += ["<{}>".format(a) for a in v]
            enumeration_by_group[tuple(clusters)] += 1

    vocabulary = list(set(vocabulary))

    return full_index, vocabulary


def construct_indices_from_cluster(args):
    if not os.path.isfile(
        args.data_dir
        + args.task
        + "/CF_indices/computed_{}_{}_CF_index.json".format(
            args.cluster_number, args.cluster_size
        )
    ):
        with open(
            args.data_dir
            + args.task
            + "/CF_indices/c{}_{}_CF_index.json".format(
                args.cluster_number, args.cluster_size
            ),
            "r",
        ) as f:
            data = json.load(f)

        mapping, vocabulary = compute_index(data)

        with open(
            args.data_dir
            + args.task
            + "/CF_indices/computed_{}_{}_CF_index.json".format(
                args.cluster_number, args.cluster_size
            ),
            "w",
        ) as f:
            json.dump([mapping, vocabulary], f)
    else:
        if not args.last_token_no_repetition:
            with open(
                args.data_dir
                + args.task
                + "/CF_indices/computed_{}_{}_CF_index.json".format(
                    args.cluster_number, args.cluster_size
                ),
                "r",
            ) as f:
                result = json.load(f)
                mapping = result[0]
                vocabulary = result[1]
        else:
            with open(
                args.data_dir
                + args.task
                + "/CF_indices/computed_no_repetition_{}_{}_CF_index.json".format(
                    args.cluster_number, args.cluster_size
                ),
                "r",
            ) as f:
                result = json.load(f)
                mapping = result[0]
                vocabulary = result[1]

    for i in range(len(vocabulary) - 1, args.number_of_items):
        vocabulary.append("<A{}>".format(i))

    return mapping, vocabulary


def construct_indices_from_cluster_optimal_width(args):
    if args.category_no_repetition:
        with open(
            args.data_dir
            + args.task
            + "/CF_indices/computed_no_repetition_optimal_{}_CF_index.json".format(
                args.cluster_size
            ),
            "r",
        ) as f:
            clustering, vocabulary = json.load(f)
        vocabulary = ["<A{}>".format(item) for item in vocabulary]
    elif args.last_token_no_repetition:
        with open(
            args.data_dir
            + args.task
            + "/CF_indices/computed_no_repetition_at_all_optimal_{}_CF_index.json".format(
                args.cluster_size
            ),
            "r",
        ) as f:
            clustering, vocabulary = json.load(f)
        vocabulary = ["<A{}>".format(item) for item in vocabulary]
    else:
        with open(
            args.data_dir
            + args.task
            + "/CF_indices/computed_optimal_{}_CF_index.json".format(args.cluster_size),
            "r",
        ) as f:
            clustering = json.load(f)
        vocabulary = ["<A{}>".format(i) for i in range(args.cluster_size)]

    CF_mapping = {
        k: "".join(["<A{}>".format(item) for item in v]) for k, v in clustering.items()
    }

    items_not_in_train = max(list([int(k) for k in CF_mapping.keys()]))
    for i in range(
        items_not_in_train, args.number_of_items
    ):  # the rest of the test/validation datapoints does not occur in train
        vocabulary.append("<A{}>".format(i))

    return CF_mapping, vocabulary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--task", type=str, default="toys")
    parser.add_argument("--cluster_size", type=int, default=200)
    parser.add_argument("--cluster_number", type=int, default=10)

    parser.add_argument("--last_token_no_repetition", action="store_true")
    parser.add_argument(
        "--category_no_repetition", action="store_true", help="usually not used though"
    )

    args = parser.parse_args()

    if args.task == "beauty":
        args.number_of_items = 12102
    elif args.task == "toys":
        args.number_of_items = 11925
    elif args.task == "sports":
        args.number_of_items = 18358
    else:
        args.number_of_items = 0

    # mapping, vocabulary = construct_indices_from_cluster(args)

    mapping, vocabulary = construct_indices_from_cluster_optimal_width(args)

    print(mapping["0"])
    print(mapping["1"])
    print(mapping["2"])
    print(mapping["3"])
    print(mapping["4"])
    print(mapping["5"])
    print(mapping["6"])
    print(mapping["7"])
    print(mapping["8"])
    print(mapping["9"])
    print(mapping["10"])
    print(mapping["11"])
    print(mapping["12"])
    print(mapping["13"])
    print(mapping["14"])
    print(mapping["15"])

    A = list(range(0, len(mapping)))
    print(len(A))
    B = sorted([int(a) for a in list(mapping.keys())])
    print(len(B))
    print(A == B)
