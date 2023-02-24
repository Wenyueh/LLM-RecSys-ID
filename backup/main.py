import argparse
from transformers import AutoTokenizer
from data import load_train_dataloaders, load_eval_dataloaders, load_data, meta_loader
from tqdm import tqdm
from utils import (
    set_seed,
    Logger,
    create_optimizer_and_scheduler,
    exact_match,
    prefix_allowed_tokens_fn,
    load_model,
    random_initialization,
)
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

transformers.logging.set_verbosity_error()


def pretrain(args, logger, rank, model, pretrain_loaders):
    if rank == 0:
        logger.log("start pretraining using meta data")
    model.zero_grad()
    pretrain_logging_loss = 0
    pretrain_logging_step = 0

    if args.use_review_data:
        num_tasks = 5
        pretrain_batch_per_epoch = (
            len(pretrain_loaders[0])
            + len(pretrain_loaders[1])
            + len(pretrain_loaders[2])
            + len(pretrain_loaders[3])
            + len(pretrain_loaders[4])
        )
        pretrain_iterators = [iter(loader) for loader in pretrain_loaders[:num_tasks]]
        pretrain_optimizer, pretrain_scheduler = create_optimizer_and_scheduler(
            args, logger, model, pretrain_batch_per_epoch, pretrain=True
        )
    else:
        num_tasks = 3
        pretrain_batch_per_epoch = (
            len(pretrain_loaders[0])
            + len(pretrain_loaders[1])
            + len(pretrain_loaders[2])
        )
        pretrain_iterators = [iter(loader) for loader in pretrain_loaders[:num_tasks]]
        pretrain_optimizer, pretrain_scheduler = create_optimizer_and_scheduler(
            args, logger, model, pretrain_batch_per_epoch, pretrain=True
        )

    for pretrain_epoch in range(args.pretrain_epochs):
        if rank == 0:
            logger.log(
                "---------- pretraining epoch {} ----------".format(pretrain_epoch)
            )
        if args.distributed:
            for loader in pretrain_loaders:
                loader.sampler.set_epoch(pretrain_epoch)
        model.train()
        for i in tqdm(range(pretrain_batch_per_epoch)):
            for j in range(num_tasks):
                if (i + 1) % num_tasks == j:
                    try:
                        batch = next(pretrain_iterators[j])
                    except StopIteration as e:
                        pretrain_iterators[j] = iter(pretrain_loaders[j])
                        batch = next(pretrain_iterators[j])

            input_ids = batch[0].to(args.gpu)
            attn = batch[1].to(args.gpu)
            whole_input_ids = batch[2].to(args.gpu)
            output_ids = batch[3].to(args.gpu)
            output_attention = batch[4].to(args.gpu)

            if args.distributed:
                output = model.module(
                    input_ids=input_ids,
                    whole_word_ids=whole_input_ids,
                    attention_mask=attn,
                    labels=output_ids,
                    alpha=args.alpha,
                    return_dict=True,
                )
            else:
                output = model(
                    input_ids=input_ids,
                    whole_word_ids=whole_input_ids,
                    attention_mask=attn,
                    labels=output_ids,
                    alpha=args.alpha,
                    return_dict=True,
                )

            # compute loss masking padded tokens
            loss = output["loss"]
            lm_mask = output_attention != 0
            lm_mask = lm_mask.float()
            B, L = output_ids.size()
            loss = loss.view(B, L) * lm_mask
            loss = (loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)).mean()

            pretrain_logging_loss += loss.item()

            # update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            pretrain_optimizer.step()
            pretrain_scheduler.step()
            model.zero_grad()

            pretrain_logging_step += 1

            if pretrain_logging_step % args.logging_step == 0 and rank == 0:
                logger.log(
                    "total loss for {} steps : {}".format(
                        pretrain_logging_step, pretrain_logging_loss
                    )
                )
                pretrain_logging_loss = 0

    if rank == 0:
        logger.log("-------finish pretraining using meta data-------")

    return model


def trainer(
    args,
    rank,
    train_loaders,
    val_loader,
    test_loader,
    pretrain_loaders,
    remapped_all_items,
    batch_per_epoch,
    tokenizer,
    logger,
):
    if rank == 0:
        logger.log("loading model ...")
        logger.log("using only sequential data, and all possible sequences are here")
    config = T5Config.from_pretrained(args.model_type)
    if args.no_pretrain:
        if rank == 0:
            logger.log("do not use pretrained T5")
        model = P5(config=config)
    else:
        if rank == 0:
            logger.log("use pretrained T5")
        model = P5.from_pretrained(args.model_type, config=config)  # .to(args.gpu)
    if args.random_initialization_embedding:
        if rank == 0:
            logger.log("randomly initialize item-related embeddings only")
        model = random_initialization(model, tokenizer)

    model.resize_token_embeddings(len(tokenizer))
    if args.eval_only:
        if os.path.isfile("best_" + args.model_dir):
            if rank == 0:
                logger.log("trained model exists and we start from this")
            model = load_model(model, "best_" + args.model_dir, rank)
    model.to(args.gpu)
    # model = P5(config).to(args.gpu)
    # if rank == 0:
    #    logger.log("finished building model")
    # if os.path.isfile(args.model_dir):
    #    logger.log("load pretrained model")
    # configure map_location properly
    #    model = load_model(model, "good" + args.model_dir, rank)

    if args.distributed:
        dist.barrier()

    if args.multiGPU:
        if rank == 0:
            logger.log("model dataparallel set")
        if args.distributed:
            model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)

    if args.use_meta_data:
        model = pretrain(args, logger, rank, model, pretrain_loaders,)

    optimizer, scheduler = create_optimizer_and_scheduler(
        args, logger, model, batch_per_epoch
    )
    if rank == 0:
        logger.log("start training")
    model.zero_grad()
    logging_step = 0
    logging_loss = 0
    best_validation_recall = 0

    number_of_tasks = 1
    train_sequential_item_dataloader = train_loaders[0]

    print("train")
    for i, batch in enumerate(tqdm(train_sequential_item_dataloader)):
        input_ids = batch[0].tolist()
        output_ids = batch[-2].tolist()
        decoded_input = tokenizer.batch_decode(input_ids)  # , skip_special_tokens=True)
        decode_ouput = tokenizer.batch_decode(output_ids)  # , skip_special_tokens=True)
        print(decoded_input[0].replace("<pad>", "").replace("</s>", ""))
        print(decode_ouput[0].replace("<pad>", "").replace("</s>", ""))
        break
    print("val")
    for batch in tqdm(val_loader):
        input_ids = batch[0].tolist()
        output_ids = batch[-2].tolist()
        decoded_input = tokenizer.batch_decode(input_ids)  # , skip_special_tokens=True)
        decode_ouput = tokenizer.batch_decode(output_ids)  # , skip_special_tokens=True)
        print(decoded_input[0].replace("<pad>", "").replace("</s>", ""))
        print(decode_ouput[0].replace("<pad>", "").replace("</s>", ""))
        break
    print("test")
    for batch in tqdm(test_loader):
        input_ids = batch[0].tolist()
        output_ids = batch[-2].tolist()
        decoded_input = tokenizer.batch_decode(input_ids)  # , skip_special_tokens=True)
        decode_ouput = tokenizer.batch_decode(output_ids)  # , skip_special_tokens=True)
        print(decoded_input[0].replace("<pad>", "").replace("</s>", ""))
        print(decode_ouput[0].replace("<pad>", "").replace("</s>", ""))
        break

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

            # for i in tqdm(range(10)):
            #    batch = next(sequential_item_iterator)
            for batch in tqdm(train_sequential_item_dataloader):
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

                if args.distributed:
                    output = model.module(
                        input_ids=input_ids,
                        whole_word_ids=whole_input_ids,
                        attention_mask=attn,
                        labels=output_ids,
                        alpha=args.alpha,
                        return_dict=True,
                    )
                else:
                    output = model(
                        input_ids=input_ids,
                        whole_word_ids=whole_input_ids,
                        attention_mask=attn,
                        labels=output_ids,
                        alpha=args.alpha,
                        return_dict=True,
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

            """
            if rank == 0:
                logger.log("---------- save model ----------")
            if args.distributed:
                torch.save(model.module.state_dict(), args.model_dir)
            else:
                torch.save(model.state_dict(), args.model_dir)
            """

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
        correct_test_1 = 0
        correct_test_5 = 0
        correct_test_10 = 0
        validation_total = 0
        test_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                input_ids = batch[0].to(args.gpu)
                attn = batch[1].to(args.gpu)
                whole_input_ids = batch[2].to(args.gpu)
                output_ids = batch[3].to(args.gpu)
                output_attention = batch[4].to(args.gpu)

                # k = 1
                if args.distributed:
                    prediction = model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=8,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    )
                else:
                    prediction = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=8,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    )
                if args.item_representation not in [
                    "no_tokenization",
                    "item_resolution",
                ]:
                    gold_sents = tokenizer.batch_decode(
                        output_ids, skip_special_tokens=True
                    )
                    generated_sents = tokenizer.batch_decode(
                        prediction, skip_special_tokens=True
                    )
                else:
                    gold_sents = [
                        a.replace("<pad>", "").replace("</s>", "").replace(" ", "")
                        for a in tokenizer.batch_decode(output_ids)
                    ]
                    generated_sents = [
                        a.replace("<pad>", "").replace("</s>", "").replace(" ", "")
                        for a in tokenizer.batch_decode(prediction)
                    ]
                correct_validation_1 += exact_match(generated_sents, gold_sents, 1)

                # k = 5
                if args.distributed:
                    prediction = model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=8,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=5,
                        num_return_sequences=5,
                    )
                else:
                    prediction = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=8,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=5,
                        num_return_sequences=5,
                    )
                if args.item_representation not in [
                    "no_tokenization",
                    "item_resolution",
                ]:
                    gold_sents = tokenizer.batch_decode(
                        output_ids, skip_special_tokens=True
                    )
                    generated_sents = tokenizer.batch_decode(
                        prediction, skip_special_tokens=True
                    )
                else:
                    gold_sents = [
                        a.replace("<pad>", "").replace("</s>", "").replace(" ", "")
                        for a in tokenizer.batch_decode(output_ids)
                    ]
                    generated_sents = [
                        a.replace("<pad>", "").replace("</s>", "").replace(" ", "")
                        for a in tokenizer.batch_decode(prediction)
                    ]
                correct_validation_5 += exact_match(generated_sents, gold_sents, 5)

                # k = 10
                if args.distributed:
                    prediction = model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=8,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=10,
                        num_return_sequences=10,
                    )
                else:
                    prediction = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=8,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=10,
                        num_return_sequences=10,
                    )
                if args.item_representation not in [
                    "no_tokenization",
                    "item_resolution",
                ]:
                    gold_sents = tokenizer.batch_decode(
                        output_ids, skip_special_tokens=True
                    )
                    generated_sents = tokenizer.batch_decode(
                        prediction, skip_special_tokens=True
                    )
                else:
                    gold_sents = [
                        a.replace("<pad>", "").replace("</s>", "").replace(" ", "")
                        for a in tokenizer.batch_decode(output_ids)
                    ]
                    generated_sents = [
                        a.replace("<pad>", "").replace("</s>", "").replace(" ", "")
                        for a in tokenizer.batch_decode(prediction)
                    ]
                correct_validation_10 += exact_match(generated_sents, gold_sents, 10)

                validation_total += len(gold_sents)

            for batch in tqdm(test_loader):
                input_ids = batch[0].to(args.gpu)
                attn = batch[1].to(args.gpu)
                whole_input_ids = batch[2].to(args.gpu)
                output_ids = batch[3].to(args.gpu)
                output_attention = batch[4].to(args.gpu)

                # k = 1
                if args.distributed:
                    prediction = model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=8,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    )
                else:
                    prediction = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=8,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    )
                if args.item_representation not in [
                    "no_tokenization",
                    "item_resolution",
                ]:
                    gold_sents = tokenizer.batch_decode(
                        output_ids, skip_special_tokens=True
                    )
                    generated_sents = tokenizer.batch_decode(
                        prediction, skip_special_tokens=True
                    )
                else:
                    gold_sents = [
                        a.replace("<pad>", "").replace("</s>", "").replace(" ", "")
                        for a in tokenizer.batch_decode(output_ids)
                    ]
                    generated_sents = [
                        a.replace("<pad>", "").replace("</s>", "").replace(" ", "")
                        for a in tokenizer.batch_decode(prediction)
                    ]
                correct_test_1 += exact_match(generated_sents, gold_sents, 1)

                # k = 5
                if args.distributed:
                    prediction = model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=8,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=5,
                        num_return_sequences=5,
                    )
                else:
                    prediction = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=8,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=5,
                        num_return_sequences=5,
                    )
                if args.item_representation not in [
                    "no_tokenization",
                    "item_resolution",
                ]:
                    gold_sents = tokenizer.batch_decode(
                        output_ids, skip_special_tokens=True
                    )
                    generated_sents = tokenizer.batch_decode(
                        prediction, skip_special_tokens=True
                    )
                else:
                    gold_sents = [
                        a.replace("<pad>", "").replace("</s>", "").replace(" ", "")
                        for a in tokenizer.batch_decode(output_ids)
                    ]
                    generated_sents = [
                        a.replace("<pad>", "").replace("</s>", "").replace(" ", "")
                        for a in tokenizer.batch_decode(prediction)
                    ]
                correct_test_5 += exact_match(generated_sents, gold_sents, 5)

                # k = 10
                if args.distributed:
                    prediction = model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=8,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=10,
                        num_return_sequences=10,
                    )
                else:
                    prediction = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=8,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=10,
                        num_return_sequences=10,
                    )
                if args.item_representation not in [
                    "no_tokenization",
                    "item_resolution",
                ]:
                    gold_sents = tokenizer.batch_decode(
                        output_ids, skip_special_tokens=True
                    )
                    generated_sents = tokenizer.batch_decode(
                        prediction, skip_special_tokens=True
                    )
                else:
                    gold_sents = [
                        a.replace("<pad>", "").replace("</s>", "").replace(" ", "")
                        for a in tokenizer.batch_decode(output_ids)
                    ]
                    generated_sents = [
                        a.replace("<pad>", "").replace("</s>", "").replace(" ", "")
                        for a in tokenizer.batch_decode(prediction)
                    ]
                correct_test_10 += exact_match(generated_sents, gold_sents, 10)

                test_total += len(gold_sents)

        recall_validation_1 = correct_validation_1 / validation_total
        recall_validation_5 = correct_validation_5 / validation_total
        recall_validation_10 = correct_validation_10 / validation_total
        recall_test_1 = correct_test_1 / test_total
        recall_test_5 = correct_test_5 / test_total
        recall_test_10 = correct_test_10 / test_total
        logger.log("validation hit @ 1 is {}".format(recall_validation_1))
        logger.log("validation hit @ 5 is {}".format(recall_validation_5))
        logger.log("validation hit @ 10 is {}".format(recall_validation_10))
        logger.log("test hit @ 1 is {}".format(recall_test_1))
        logger.log("test hit @ 5 is {}".format(recall_test_5))
        logger.log("test hit @ 10 is {}".format(recall_test_10))

        if recall_validation_1 > best_validation_recall:
            model_dir = "best_" + args.model_dir
            logger.log(
                "recall increases from {} ----> {} at epoch {}".format(
                    best_validation_recall, recall_validation_1, epoch
                )
            )
            torch.save(model.state_dict(), model_dir)
            best_validation_recall = recall_validation_1

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
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    if args.item_representation == "no_tokenization":
        new_tokens = []
        for x in range(12102):
            new_token = "<extra_id_{}>".format(x)
            new_tokens.append(new_token)
        new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
        tokenizer.add_tokens(list(new_tokens))
    elif args.item_representation == "item_resolution":
        new_tokens = []
        number_of_new_tokens = min(10 ** args.resolution, 12102)
        for x in range(number_of_new_tokens):
            new_token = "<extra_id_{}>".format(x)
            new_tokens.append(new_token)
        new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
        tokenizer.add_tokens(list(new_tokens))
    if args.user_representation == "no_tokenization":
        new_tokens = []
        for x in range(22364):
            new_token = "<user_id_{}>".format(x)
            new_tokens.append(new_token)
        new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
        tokenizer.add_tokens(list(new_tokens))

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
    """
    print("validation")
    for batch in val_loader:
        input_ids = batch[0].tolist()
        print(tokenizer.batch_decode(input_ids[:4]))
        time.sleep(5)
    """

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
    """
    print("test")
    for batch in test_loader:
        input_ids = batch[0].tolist()
        print(tokenizer.batch_decode(input_ids[:4]))
        time.sleep(5)
    """

    # pretrain using meta data
    if args.use_meta_data:
        (
            pretrain_title_dataloader,
            pretrain_description_identification_dataloader,
            pretrain_category_datalaoder,
            pretrain_review_item_dataloader,
            pretrain_review_user_dataloader,
        ) = meta_loader(args, tokenizer, all_items, remapped_all_items)
        pretrain_loaders = [
            pretrain_title_dataloader,
            pretrain_description_identification_dataloader,
            pretrain_category_datalaoder,
            pretrain_review_item_dataloader,
            pretrain_review_user_dataloader,
        ]
    else:
        pretrain_loaders = None

    trainer(
        args,
        local_rank,
        train_loaders,
        val_loader,
        test_loader,
        pretrain_loaders,
        remapped_all_items,
        batch_per_epoch,
        tokenizer,
        logger,
    )


def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--task", type=str, default="beauty")

    parser.add_argument("--max_history", type=int, default=20)
    parser.add_argument("--sequential_num", type=int, default=10)
    parser.add_argument("--negative_sample", type=int, default=2)
    parser.add_argument("--yes_no_sample", type=int, default=5)
    parser.add_argument("--direct_item_proportion", type=int, default=2)

    parser.add_argument("--train_sequential_item_batch", type=int, default=128)
    parser.add_argument("--train_sequential_yesno_batch", type=int, default=48)
    parser.add_argument("--train_direct_yesno_batch", type=int, default=48)
    parser.add_argument("--train_direct_candidate_batch", type=int, default=12)
    parser.add_argument("--train_direct_straightforward_batch", type=int, default=48)

    parser.add_argument("--pretrain_title_batch", type=int, default=128)
    parser.add_argument("--pretrain_description_batch", type=int, default=12)
    parser.add_argument("--pretrain_category_batch", type=int, default=128)
    parser.add_argument("--pretrain_review_user_batch", type=int, default=12)
    parser.add_argument("--pretrain_review_item_batch", type=int, default=12)
    parser.add_argument("--pretrain_epochs", type=int, default=5)

    parser.add_argument("--logging_dir", type=str, default="beauty.log")
    parser.add_argument("--model_dir", type=str, default="pretrain_t5_small_beauty.pt")

    # learning hyperparameters
    parser.add_argument("--model_type", type=str, default="t5-small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
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

    #### experiment setting ####
    parser.add_argument(
        "--item_representation",
        type=str,
        default="random_number",
        help="random_number, random_one_token, item_resolution, no_tokenization, None",
    )
    parser.add_argument(
        "--number_base",
        type=int,
        default=10,
        help="change the base for number representation",
    )
    parser.add_argument(
        "--user_representation",
        type=str,
        default="None",
        help="no tokenization or None",
    )
    parser.add_argument(
        "--resolution", type=int, default=2, help="from 1 to 5 for beauty"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="0 is no overlap, overlap must < resolution",
    )
    parser.add_argument(
        "--random_initialization_embedding",
        action="store_true",
        help="randomly initialize number related tokens",
    )
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
        help="use review data identification tasks to pretrain the model",
    )
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

    args = parser.parse_args()

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
