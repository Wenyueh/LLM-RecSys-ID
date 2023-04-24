from sklearn.cluster import SpectralClustering
import argparse
from transformers import AutoTokenizer
from data import load_train_dataloaders, load_eval_dataloaders, load_data
from pretrain_data import meta_loader, review_loader
from tqdm import tqdm
from utils import (
    set_seed,
    Logger,
    create_optimizer_and_scheduler,
    exact_match,
    prefix_allowed_tokens_fn,
    load_model,
    random_initialization,
    create_category_embedding,
    create_category_embedding_yelp,
    content_category_embedding_modified_yelp,
    content_based_representation_non_hierarchical,
)
from item_rep_method import (
    create_CF_embedding,
    create_hybrid_embedding,
    load_hybrid,
    create_CF_embedding_optimal_width,
    build_category_map,
    load_meta,
    build_category_map_modified_yelp,
)
from pretrain_main import pretrain_using_meta_data, pretrain_using_review_data
from modeling_p5 import P5
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import T5Config
import transformers
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import time
from generation_trie import Trie
import os
from collections import OrderedDict

transformers.logging.set_verbosity_error()

import warnings

warnings.filterwarnings("ignore")


def predict_outputs(args, batch, model, tokenizer, prefix_allowed_tokens, k=20):
    input_ids = batch[0].to(args.gpu)
    attn = batch[1].to(args.gpu)
    whole_input_ids = batch[2].to(args.gpu)
    output_ids = batch[3].to(args.gpu)

    if args.whole_word_embedding == "None":
        if args.distributed:
            prediction = model.module.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_length=8,
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                num_beams=20,
                num_return_sequences=20,
                whole_word_embedding_type=args.whole_word_embedding,
                output_scores=True,
                return_dict_in_generate=True,
            )
        else:
            prediction = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_length=8,
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                num_beams=20,
                num_return_sequences=20,
                whole_word_embedding_type=args.whole_word_embedding,
                output_scores=True,
                return_dict_in_generate=True,
            )
    else:
        # k = 1
        if args.distributed:
            prediction = model.module.generate(
                input_ids=input_ids,
                attention_mask=attn,
                whole_word_ids=whole_input_ids,
                max_length=8,
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                num_beams=20,
                num_return_sequences=20,
                whole_word_embedding_type=args.whole_word_embedding,
                output_scores=True,
                return_dict_in_generate=True,
            )
        else:
            prediction = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                whole_word_ids=whole_input_ids,
                max_length=8,
                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                num_beams=20,
                num_return_sequences=20,
                whole_word_embedding_type=args.whole_word_embedding,
                output_scores=True,
                return_dict_in_generate=True,
            )

    prediction_ids = prediction["sequences"]
    prediction_scores = prediction["sequences_scores"]

    if args.item_representation not in [
        "no_tokenization",
        "item_resolution",
    ]:
        gold_sents = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        generated_sents = tokenizer.batch_decode(
            prediction_ids, skip_special_tokens=True
        )
    else:
        gold_sents = [
            a.replace("<pad>", "").replace("</s>", "").replace(" ", "")
            for a in tokenizer.batch_decode(output_ids)
        ]
        generated_sents = [
            a.replace("<pad>", "").replace("</s>", "").replace(" ", "")
            for a in tokenizer.batch_decode(prediction_ids)
        ]
    hit_1, hit_5, hit_10, ncdg_5, ncdg_10 = exact_match(
        generated_sents, gold_sents, prediction_scores, 20
    )

    return hit_1, hit_5, hit_10, ncdg_5, ncdg_10


def trainer(
    args,
    rank,
    train_loaders,
    val_loader,
    test_loader,
    pretrain_loaders,
    review_loaders,
    remapped_all_items,
    batch_per_epoch,
    tokenizer,
    logger,
):

    if rank == 0:
        logger.log("loading model ...")
        logger.log("using only sequential data, and all possible sequences are here")
    config = T5Config.from_pretrained(args.model_type)
    config.dropout_rate = args.dropout
    if args.no_pretrain:
        if rank == 0:
            logger.log("do not use pretrained T5")
        model = P5(config=config)
    else:
        if rank == 0:
            logger.log("use pretrained T5")
        model = P5.from_pretrained(
            pretrained_model_name_or_path=args.model_type,
            config=config,
            # **model_args,  # , args=args
        )  # .to(args.gpu)
    if args.random_initialization_embedding:
        if rank == 0:
            logger.log("randomly initialize number-related embeddings only")
        model = random_initialization(model, tokenizer)

    model.resize_token_embeddings(len(tokenizer))
    model.to(args.gpu)

    optimizer, scheduler = create_optimizer_and_scheduler(
        args, logger, model, batch_per_epoch
    )

    if args.distributed:
        dist.barrier()

    if args.multiGPU:
        if rank == 0:
            logger.log("model dataparallel set")
        if args.distributed:
            model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)

    """
    if args.eval_only:
        if os.path.isfile("best_" + args.model_dir):
            if rank == 0:
                logger.log("trained model exists and we start from this")
            # model = load_model(model, "best_" + args.model_dir, rank)
            map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
            model.load_state_dict(
                torch.load("best_" + args.model_dir, map_location=map_location)
            )
    """

    if args.use_meta_data:
        model = pretrain_using_meta_data(args, logger, rank, model, pretrain_loaders)
    if args.use_review_data:
        model = pretrain_using_review_data(args, logger, rank, model, review_loaders)

    if rank == 0:
        logger.log("start training")
    model.zero_grad()
    logging_step = 0
    logging_loss = 0
    best_validation_recall = 0

    number_of_tasks = 1

    if args.check_data:

        train_sequential_item_dataloader = train_loaders[0]
        print("********check train data********")
        for batch in train_sequential_item_dataloader:
            one = batch[0]
            answers = batch[3]
            decoded = tokenizer.batch_decode(one)
            decoded_answers = tokenizer.batch_decode(answers)
            for a, b in zip(decoded, decoded_answers):
                if "user_1 " in a:
                    logger.log("rank is {}".format(rank))
                    print(tokenizer.convert_ids_to_tokens(one[0]))
                    print((a, b))
                    print("***")
                    time.sleep(30)

        print("********check val data********")
        for batch in val_loader:
            one = batch[0]
            answers = batch[3]
            decoded = tokenizer.batch_decode(one)
            decoded_answers = tokenizer.batch_decode(answers)
            for a, b in zip(decoded, decoded_answers):
                if "User_1 " in a:
                    logger.log("rank is {}".format(rank))
                    print((a, b))
                    print("***")
                    time.sleep(10)

    # train_sequential_yesno_dataloader = train_loaders[1]
    # train_direct_yesno_dataloader = train_loaders[2]
    # train_direct_candidate_dataloader = train_loaders[3]
    # train_direct_straightforward_dataloader = train_loaders[4]

    for epoch in range(args.epochs):
        if rank == 0:
            logger.log("---------- training epoch {} ----------".format(epoch))
        if args.distributed:
            for loader in train_loaders:
                loader.sampler.set_epoch(epoch)

        if not args.eval_only:
            model.train()

            # sequential_item_iterator = iter(train_sequential_item_dataloader)
            # sequential_yesno_iterator = iter(train_sequential_yesno_dataloader)
            # direct_yesno_iterator = iter(train_direct_yesno_dataloader)
            # direct_candidate_iterator = iter(train_direct_candidate_dataloader)
            # direct_straightforward_iterator = iter(train_direct_straightforward_dataloader)

            for batch in tqdm(train_loaders[0]):
                """
                for i in tqdm(range(batch_per_epoch)):
                batch = next(sequential_item_iterator)

                # for i in tqdm(range(100)):
                if (i + 1) % number_of_tasks == 1:
                    try:
                        batch = next(sequential_item_iterator)
                    except StopIteration as e:
                        sequential_item_iterator = iter(train_sequential_item_dataloader)
                        batch = next(sequential_item_iterator)
                elif (i + 1) % number_of_tasks == 0:
                    try:
                        batch = next(sequential_yesno_iterator)
                    except StopIteration as e:
                        sequential_yesno_iterator = iter(train_sequential_yesno_dataloader)
                        batch = next(sequential_yesno_iterator)
                elif (i + 1) % number_of_tasks == 3:
                    try:
                        batch = next(direct_yesno_iterator)
                    except StopIteration as e:
                        direct_yesno_iterator = iter(train_direct_yesno_dataloader)
                        batch = next(direct_yesno_iterator)
                elif (i + 1) % number_of_tasks == 4:
                    try:
                        batch = next(direct_candidate_iterator)
                    except StopIteration as e:
                        direct_candidate_iterator = iter(train_direct_candidate_dataloader)
                        batch = next(direct_candidate_iterator)
                else:
                    assert (i + 1) % number_of_tasks == 0
                    try:
                        batch = next(direct_straightforward_iterator)
                    except StopIteration as e:
                        direct_straightforward_iterator = iter(
                            train_direct_straightforward_dataloader
                        )
                        batch = next(direct_straightforward_iterator)
                """

                input_ids = batch[0].to(args.gpu)
                attn = batch[1].to(args.gpu)
                whole_input_ids = batch[2].to(args.gpu)
                output_ids = batch[3].to(args.gpu)
                output_attention = batch[4].to(args.gpu)

                if args.whole_word_embedding == "None":
                    if args.distributed:
                        output = model.module(
                            input_ids=input_ids,
                            attention_mask=attn,
                            labels=output_ids,
                            alpha=args.alpha,
                            return_dict=True,
                            whole_word_embedding_type=args.whole_word_embedding,
                        )
                    else:
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attn,
                            labels=output_ids,
                            alpha=args.alpha,
                            return_dict=True,
                            whole_word_embedding_type=args.whole_word_embedding,
                        )
                else:
                    if args.distributed:
                        output = model.module(
                            input_ids=input_ids,
                            whole_word_ids=whole_input_ids,
                            attention_mask=attn,
                            labels=output_ids,
                            alpha=args.alpha,
                            return_dict=True,
                            whole_word_embedding_type=args.whole_word_embedding,
                        )
                    else:
                        output = model(
                            input_ids=input_ids,
                            whole_word_ids=whole_input_ids,
                            attention_mask=attn,
                            labels=output_ids,
                            alpha=args.alpha,
                            return_dict=True,
                            whole_word_embedding_type=args.whole_word_embedding,
                        )

                # compute loss masking padded tokens
                loss = output["loss"]
                lm_mask = output_attention != 0
                lm_mask = lm_mask.float()
                B, L = output_ids.size()
                loss = loss.view(B, L) * lm_mask
                loss = (loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()

                logging_loss += loss.item()

                # update
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                logging_step += 1

                if logging_step % args.logging_step == 0 and rank == 0:
                    logger.log(
                        "total loss for {} steps : {}".format(
                            logging_step, logging_loss
                        )
                    )
                    logging_loss = 0

            dist.barrier()

        if rank == 0:
            logger.log(
                "---------- start evaluation after epoch {} ----------".format(epoch)
            )
        if args.evaluation_method == "sequential_item":
            candidates = remapped_all_items
            candidate_trie = Trie(
                [
                    [0] + tokenizer.encode("{}".format("item_" + candidate))
                    for candidate in candidates
                ]
            )
        elif args.evaluation_method == "sequential_yesno":
            candidates = ["yes", "no"]
            candidate_trie = Trie(
                [
                    [0] + tokenizer.encode("{}".format(candidate))
                    for candidate in candidates
                ]
            )
        elif args.evaluation_method == "direct_yesno":
            candidates = ["yes", "no"]
            candidate_trie = Trie(
                [
                    [0] + tokenizer.encode("{}".format(candidate))
                    for candidate in candidates
                ]
            )
        elif args.evaluation_method == "direct_candidate":
            candidates = remapped_all_items
            candidate_trie = Trie(
                [
                    [0] + tokenizer.encode("{}".format("item_" + candidate))
                    for candidate in candidates
                ]
            )
        prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

        model.eval()
        correct_validation_1 = 0
        correct_validation_5 = 0
        correct_validation_10 = 0
        ncdg_validation_5 = 0
        ncdg_validation_10 = 0
        correct_test_1 = 0
        correct_test_5 = 0
        correct_test_10 = 0
        ncdg_test_5 = 0
        ncdg_test_10 = 0
        validation_total = 0
        test_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                output_ids = batch[3]

                (
                    one_hit_1,
                    one_hit_5,
                    one_hit_10,
                    one_ncdg_5,
                    one_ncdg_10,
                ) = predict_outputs(
                    args, batch, model, tokenizer, prefix_allowed_tokens,
                )

                correct_validation_1 += one_hit_1
                correct_validation_5 += one_hit_5
                correct_validation_10 += one_hit_10
                ncdg_validation_5 += one_ncdg_5
                ncdg_validation_10 += one_ncdg_10

                validation_total += output_ids.size(0)

            recall_validation_1 = correct_validation_1 / validation_total
            recall_validation_5 = correct_validation_5 / validation_total
            recall_validation_10 = correct_validation_10 / validation_total
            ncdg_validation_5 = ncdg_validation_5 / validation_total
            ncdg_validation_10 = ncdg_validation_10 / validation_total
            logger.log("validation hit @ 1 is {}".format(recall_validation_1))
            logger.log("validation hit @ 5 is {}".format(recall_validation_5))
            logger.log("validation hit @ 10 is {}".format(recall_validation_10))
            logger.log("validation ncdg @ 5 is {}".format(ncdg_validation_5))
            logger.log("validation ncdg @ 10 is {}".format(ncdg_validation_10))

            for batch in tqdm(test_loader):
                output_ids = batch[3].to(args.gpu)

                (
                    one_hit_1,
                    one_hit_5,
                    one_hit_10,
                    one_ncdg_5,
                    one_ncdg_10,
                ) = predict_outputs(
                    args, batch, model, tokenizer, prefix_allowed_tokens,
                )

                correct_test_1 += one_hit_1
                correct_test_5 += one_hit_5
                correct_test_10 += one_hit_10
                ncdg_test_5 += one_ncdg_5
                ncdg_test_10 += one_ncdg_10

                test_total += output_ids.size(0)

            recall_test_1 = correct_test_1 / test_total
            recall_test_5 = correct_test_5 / test_total
            recall_test_10 = correct_test_10 / test_total
            ncdg_test_5 = ncdg_test_5 / test_total
            ncdg_test_10 = ncdg_test_10 / test_total
            logger.log("test hit @ 1 is {}".format(recall_test_1))
            logger.log("test hit @ 5 is {}".format(recall_test_5))
            logger.log("test hit @ 10 is {}".format(recall_test_10))
            logger.log("test ncdg @ 5 is {}".format(ncdg_test_5))
            logger.log("test ncdg @ 10 is {}".format(ncdg_test_10))

        if recall_validation_10 > best_validation_recall:
            model_dir = "best_" + args.model_dir
            logger.log(
                "recall increases from {} ----> {} at epoch {}".format(
                    best_validation_recall, recall_validation_10, epoch
                )
            )
            if rank == 0:
                logger.log("save current best model to {}".format(model_dir))
                torch.save(model.module.state_dict(), model_dir)
            best_validation_recall = recall_validation_10

        dist.barrier()


def main_worker(local_rank, args, logger):
    set_seed(args)

    args.gpu = local_rank
    args.rank = local_rank
    logger.log(f"Process Launching at GPU {args.gpu}")

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(
            backend="nccl", world_size=args.world_size, rank=args.rank
        )

    logger.log(f"Building train loader at GPU {args.gpu}")

    if local_rank == 0:
        logger.log("loading data ...")

    # build tokenizers and new model embeddings
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    """
    if args.task == "beauty":
        args.number_of_items = 12102
    elif args.task == "toys":
        args.number_of_items = 11925
    elif args.task == "sports":
        args.number_of_items = 18358
    else:
        args.number_of_items = 0
    """
    number_of_items = args.number_of_items

    if args.item_representation == "no_tokenization":
        if local_rank == 0:
            logger.log(
                "*** use no tokenization setting, highest resolution, extend vocab ***"
            )
        new_tokens = []
        for x in range(number_of_items):
            new_token = "<extra_id_{}>".format(x)
            new_tokens.append(new_token)
        new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
        tokenizer.add_tokens(list(new_tokens))

    elif args.item_representation == "item_resolution":
        if local_rank == 0:
            logger.log(
                "*** use resolution = {} and overlap = {}, extend vocab ***".format(
                    args.resolution, args.overlap
                )
            )
        new_tokens = []
        number_of_new_tokens = min(10 ** args.resolution, number_of_items)
        for x in range(number_of_new_tokens):
            new_token = "<extra_id_{}>".format(x)
            new_tokens.append(new_token)
        new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
        tokenizer.add_tokens(list(new_tokens))

    elif args.item_representation == "content_based":
        if local_rank == 0:
            logger.log("*** use content_based representation, extend vocab ***")
        # if args.task != "yelp":
        #    tokenizer = create_category_embedding(args, tokenizer)
        # else:
        #    tokenizer = create_category_embedding_yelp(
        #       args, category_dict, level_categories, tokenizer
        # )
        #    category_dict = build_category_map_modified_yelp(args)
        #    tokenizer = content_category_embedding_modified_yelp(
        #       args, category_dict, tokenizer
        #    )
        if args.task == "yelp":
            tokenizer = create_category_embedding(args, tokenizer)
        else:
            meta_data, meta_dict, id2item = load_meta(args)
            category_dict, level_categories = build_category_map(
                args, meta_data, meta_dict, id2item
            )
            tokenizer = content_based_representation_non_hierarchical(
                args, category_dict, level_categories, tokenizer
            )

    elif args.item_representation == "CF":
        if local_rank == 0:
            logger.log(
                "*** use collaborative_filtering_based representation, extend vocab ***"
            )
        if not args.optimal_width_in_CF:
            tokenizer = create_CF_embedding(args, tokenizer)
        else:
            tokenizer = create_CF_embedding_optimal_width(args, tokenizer)

    elif args.item_representation == "hybrid":
        if local_rank == 0:
            logger.log(
                "*** use hybrid_based representation using metadata and CF, extend vocab ***"
            )
        _, vocabulary = load_hybrid(args)
        tokenizer = create_hybrid_embedding(vocabulary, tokenizer)

    if args.item_representation == "remapped_sequential":
        if local_rank == 0:
            logger.log("*** use remapped sequential data ***")
        assert args.random_initialization_embedding

    (
        users,
        all_items,
        train_sequence,
        val_sequence,
        test_sequence,
        remapped_all_items,
    ) = load_data(args, tokenizer)

    (
        train_sequential_item_dataloader,
        train_sequential_yesno_dataloader,
        train_direct_yesno_dataloader,
        train_direct_candidate_dataloader,
        train_direct_straightforward_dataloader,
        task_data_lengths,
        remapped_all_items,
    ) = load_train_dataloaders(
        args, tokenizer, users, remapped_all_items, train_sequence, test_sequence
    )

    batch_per_epoch = len(train_sequential_item_dataloader)  # * 2
    train_loaders = [
        train_sequential_item_dataloader,
        train_sequential_yesno_dataloader,
        train_direct_yesno_dataloader,
        train_direct_candidate_dataloader,
        train_direct_straightforward_dataloader,
    ]

    if local_rank == 0:
        logger.log("finished loading data")
        logger.log("length of training data is {}".format(batch_per_epoch))

    val_loader = load_eval_dataloaders(
        args,
        tokenizer,
        args.evaluation_method,
        "validation",
        users,
        remapped_all_items,
        val_sequence,
        test_sequence,
    )

    test_loader = load_eval_dataloaders(
        args,
        tokenizer,
        args.evaluation_method,
        "test",
        users,
        remapped_all_items,
        test_sequence,
        test_sequence,
    )

    # pretrain using meta data
    if args.use_meta_data:
        (
            pretrain_title_dataloader,
            pretrain_description_identification_dataloader,
            pretrain_category_datalaoder,
        ) = meta_loader(args, tokenizer, all_items, remapped_all_items)
        pretrain_loaders = [
            pretrain_title_dataloader,
            pretrain_description_identification_dataloader,
            pretrain_category_datalaoder,
        ]
    else:
        pretrain_loaders = None

    if args.use_review_data:
        (
            pretrain_review_item_dataloader,
            pretrain_review_user_dataloader,
        ) = review_loader(args, tokenizer, all_items, remapped_all_items)
        review_loaders = [
            pretrain_review_item_dataloader,
            pretrain_review_user_dataloader,
        ]
    else:
        review_loaders = None

    trainer(
        args,
        local_rank,
        train_loaders,
        val_loader,
        test_loader,
        pretrain_loaders,
        review_loaders,
        remapped_all_items,
        batch_per_epoch,
        tokenizer,
        logger,
    )


def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--logging_dir", type=str, default="beauty.log")
    parser.add_argument("--model_dir", type=str, default="pretrain_t5_small_beauty.pt")
    parser.add_argument("--task", type=str, default="beauty")

    # collaborative filtering setting
    parser.add_argument("--max_history", type=int, default=20)
    parser.add_argument("--sequential_num", type=int, default=10)
    parser.add_argument("--negative_sample", type=int, default=2)
    parser.add_argument("--yes_no_sample", type=int, default=5)
    parser.add_argument("--direct_item_proportion", type=int, default=2)

    # collaborative filtering batch
    parser.add_argument("--train_sequential_item_batch", type=int, default=128)
    parser.add_argument("--train_sequential_yesno_batch", type=int, default=32)
    parser.add_argument("--train_direct_yesno_batch", type=int, default=48)
    parser.add_argument("--train_direct_candidate_batch", type=int, default=12)
    parser.add_argument("--train_direct_straightforward_batch", type=int, default=48)

    # meta pretrain_related
    parser.add_argument("--meta_title_batch", type=int, default=128)
    parser.add_argument("--meta_description_batch", type=int, default=12)
    parser.add_argument("--meta_category_batch", type=int, default=128)
    parser.add_argument("--meta_epochs", type=int, default=10)
    parser.add_argument("--meta_lr", type=float, default=1e-4)

    # review pretrain_related
    parser.add_argument("--review_user_batch", type=int, default=16)
    parser.add_argument("--review_item_batch", type=int, default=16)
    parser.add_argument("--review_epochs", type=int, default=2)
    parser.add_argument("--review_lr", type=float, default=1e-4)

    # learning hyperparameters
    parser.add_argument("--model_type", type=str, default="t5-small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=1)
    parser.add_argument("--logging_step", type=int, default=100)
    parser.add_argument("--warmup_prop", type=float, default=0.05)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=2)

    # CPU/GPU
    parser.add_argument("--multiGPU", action="store_const", default=False, const=True)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("--evaluation_method", type=str, default="sequential_item")
    parser.add_argument("--evaluation_template_id", type=int, default=0)

    parser.add_argument(
        "--number_of_items",
        type=int,
        default=12102,
        help="number of items in each dataset, beauty 12102, toys 11925, sports 18358",
    )

    # item representation experiment setting
    parser.add_argument(
        "--item_representation",
        type=str,
        default="None",
        help="random_number, random_one_token, no_tokenization, item_resolution, content_based,title,CF,hybrid, None",
    )

    # arguments for collaborative indexing
    parser.add_argument(
        "--cluster_number",
        type=int,
        default=55,
        help="number of clusters to divide every step when item representation method is CF",
    )
    parser.add_argument(
        "--cluster_size",
        type=int,
        default=100,
        help="number of items in the largest clusters",
    )
    parser.add_argument(
        "--optimal_width_in_CF",
        action="store_true",
        help="whether to use eigengap heuristics to find the optimal width in CF, all repetition",
    )
    parser.add_argument(
        "--category_no_repetition",
        action="store_true",
        help="use all different tokens for non-leaf node in indexing time",
    )
    parser.add_argument(
        "--last_token_no_repetition",
        action="store_true",
        help="use all different tokens for leaf node in indexing time, collaborative + independent indexing",
    )
    parser.add_argument(
        "--hybrid_order",
        type=str,
        default="CF_first",
        help="CF_first or category_first in concatenation",
    )

    parser.add_argument(
        "--data_order",
        type=str,
        default="random",
        help="random or remapped_sequential (excluding validation and test)",
    )

    # arguments for sequential indexing
    parser.add_argument(
        "--remapped_data_order",
        type=str,
        default="original",
        help="original (original file), short_to_long, long_to_short, randomize, used when item_representation == remapped_sequential",
    )

    # arguments for random number indexing
    parser.add_argument(
        "--resolution", type=int, default=2, help="from 1 to 5 for beauty"
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
        "--overlap",
        type=int,
        default=0,
        help="0 is no overlap, overlap must < resolution",
    )
    parser.add_argument(
        "--base", type=int, default=10, help="base on number, 2,3,4",
    )

    # for None or remapped sequential
    parser.add_argument(
        "--random_initialization_embedding",
        action="store_true",
        help="randomly initialize number related tokens, use only for random_number setting, used when item_representation is None or remapped sequential",
    )

    # whether to use whole word embedding and how
    parser.add_argument(
        "--whole_word_embedding",
        type=str,
        default="shijie",
        help="shijie, None, position_embedding",
    )

    # whether to use pretrain
    parser.add_argument(
        "--no_pretrain", action="store_true", help="does not use pretrained T5 model"
    )
    parser.add_argument(
        "--use_meta_data",
        action="store_true",
        help="use meta data identification tasks to pretrain the model",
    )
    parser.add_argument(
        "--use_review_data",
        action="store_true",
        help="use user review data identification tasks to pretrain the model",
    )

    # whether modify the evaluation setting
    parser.add_argument(
        "--remove_last_item",
        action="store_true",
        help="remove last item in a sequence in test time",
    )
    parser.add_argument(
        "--remove_first_item",
        action="store_true",
        help="remove first item in a sequence in test time",
    )

    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--check_data",
        action="store_true",
        help="check whether data are correctly formated and whether consistent across GPUs",
    )

    args = parser.parse_args()

    if args.task == "beauty":
        args.number_of_items = 12102
    elif args.task == "toys":
        args.number_of_items = 11925
    elif args.task == "sports":
        args.number_of_items = 18358
    else:
        assert args.task == "yelp"
        args.number_of_items = 20034

    return args


if __name__ == "__main__":
    transformers.logging.set_verbosity_error()

    cudnn.benchmark = True
    args = parse_argument()

    set_seed(args)
    logger = Logger(args.logging_dir, True)
    logger.log(str(args))

    # number of visible gpus set in os[environ]
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    if args.distributed:
        mp.spawn(
            main_worker, args=(args, logger), nprocs=args.world_size, join=True,
        )
