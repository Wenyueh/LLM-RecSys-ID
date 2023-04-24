from utils import create_optimizer_and_scheduler
from tqdm import tqdm
import torch


def pretrain_using_meta_data(args, logger, rank, model, pretrain_loaders):
    if rank == 0:
        logger.log("start pretraining using meta data")
    model.zero_grad()
    pretrain_logging_loss = 0
    pretrain_logging_step = 0
    title_loader = pretrain_loaders[0]
    description_loader = pretrain_loaders[1]
    category_loader = pretrain_loaders[2]
    pretrain_step_per_epoch = (
        len(title_loader) + len(description_loader) + len(category_loader)
    )

    meta_optimizer, meta_scheduler = create_optimizer_and_scheduler(
        args, logger, model, pretrain_step_per_epoch, pretrain="meta"
    )
    for pretrain_epoch in range(args.meta_epochs):
        if rank == 0:
            logger.log(
                "---------- pretraining meta epoch {} ----------".format(pretrain_epoch)
            )
        if args.distributed:
            for loader in pretrain_loaders:
                loader.sampler.set_epoch(pretrain_epoch)
        model.train()
        title_iterator = iter(title_loader)
        description_iterator = iter(description_loader)
        category_iterator = iter(category_loader)
        for i in tqdm(range(pretrain_step_per_epoch)):
            if (i + 1) % 3 == 1:
                try:
                    batch = next(title_iterator)
                except StopIteration as e:
                    title_iterator = iter(title_loader)
                    batch = next(title_iterator)
            elif (i + 1) % 3 == 2:
                try:
                    batch = next(description_iterator)
                except StopIteration as e:
                    description_iterator = iter(description_loader)
                    batch = next(description_iterator)
            else:
                assert (i + 1) % 3 == 0
                try:
                    batch = next(category_iterator)
                except StopIteration as e:
                    category_iterator = iter(category_loader)
                    batch = next(category_iterator)

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
            meta_optimizer.step()
            meta_scheduler.step()
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


def pretrain_using_review_data(args, logger, rank, model, review_loaders):
    if rank == 0:
        logger.log("start pretraining using review data")
    model.zero_grad()
    pretrain_logging_loss = 0
    pretrain_logging_step = 0

    user_loader = review_loaders[0]
    item_loader = review_loaders[1]

    pretrain_step_per_epoch = len(user_loader) + len(item_loader)

    review_optimizer, review_scheduler = create_optimizer_and_scheduler(
        args, logger, model, pretrain_step_per_epoch, pretrain="review"
    )
    for pretrain_epoch in range(args.review_epochs):
        if rank == 0:
            logger.log(
                "---------- pretraining review epoch {} ----------".format(
                    pretrain_epoch
                )
            )
        if args.distributed:
            for loader in review_loaders:
                loader.sampler.set_epoch(pretrain_epoch)
        model.train()
        user_iterator = iter(user_loader)
        item_iterator = iter(item_loader)
        for i in tqdm(range(pretrain_step_per_epoch)):
            if (i + 1) % 2 == 0:
                try:
                    batch = next(user_iterator)
                except StopIteration as e:
                    user_iterator = iter(user_loader)
                    batch = next(user_iterator)
            else:
                assert (i + 1) % 2 == 1
                try:
                    batch = next(item_iterator)
                except StopIteration as e:
                    item_iterator = iter(item_loader)
                    batch = next(item_iterator)

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
            review_optimizer.step()
            review_scheduler.step()
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
        logger.log("-------finish pretraining using review data-------")

    return model
