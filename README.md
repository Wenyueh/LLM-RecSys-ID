# How to Index Item IDs for Recommendation Foundation Models

This repo presents various methods for creating item IDs for recommendation foundation models, including Random Indexing (RID), Independent Indexing (IID), Title Indexing (TID), Sequential Indexing (SID), Collaborative Indexing (CID), Semantic Indexing (SemID), and Hybrid Indexing (HID).
> Paper: How to Index Item IDs for Recommendation Foundation Models <br>
> Paper link: [https://arxiv.org/pdf/2305.06569.pdf](https://arxiv.org/pdf/2305.06569.pdf)

A relevant repo (OpenP5) of benchmarking foundation models for recommendation is also available at GitHub:
> Paper: OpenP5: Benchmarking Foundation Models for Recommendation <br>
> Paper link: [https://arxiv.org/pdf/2306.11134.pdf](https://arxiv.org/pdf/2306.11134.pdf) <br>
> GitHub link: [https://github.com/agiresearch/OpenP5](https://github.com/agiresearch/OpenP5)

# Abstract

Recommendation foundation model utilizes large language models (LLM) for recommendation by converting recommendation tasks into natural language tasks. It enables generative recommendation which directly generates the item(s) to recommend rather than calculating a ranking score for each and every candidate item as in traditional recommendation models, simplifying the recommendation pipeline from multi-stage filtering to single-stage filtering. To avoid generating excessively long text and hallucinated recommendations when deciding which item(s) to recommend, creating LLM-compatible item IDs to uniquely identify each item is essential for recommendation foundation models. In this study, we systematically examine the item ID creation and indexing problem for recommendation foundation models, using P5 as an example of the backbone LLM. To emphasize the importance of item indexing, we first discuss the issues of several trivial item indexing methods, such as random indexing, title indexing, and independent indexing. We then propose four simple yet effective solutions, including sequential indexing, collaborative indexing, semantic (content-based) indexing, and hybrid indexing. Our study highlights the significant influence of item indexing methods on the performance of LLM-based recommendation, and our results on real-world datasets validate the effectiveness of our proposed solutions. The research also demonstrates how recent advances on language modeling and traditional IR principles such as indexing can help each other for better learning and inference..

<img width="900" alt="LLM-RecSys-ID" src="image/LLM-RecSys-ID.png">

# Before running script

## Data downloading
Download meta data from here: https://drive.google.com/file/d/1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G/view, put in the data/ directory


## Environment
Build environment:
```
conda create -n itemrep -y python=3.9.7 && conda activate itemrep
pip install -r requirements.txt
```

Experiments are done on 2 or 4 A5000 GPUs with 

CUDA Driver Version: 515.86.01

CUDA Version: 11.7. 

Variations in performance may occur using different environments or GPUs. You may need to tune the hyperparameters in such a case.

# Potential bugs you would meet

If you met a bug about model_kwargs: 
```
ValueError: The following model_kwargs are not used by the model: ['whole_word_embedding_type'] (note: typos in the generate arguments will also show up in this list)
```
Check [Issue #2](https://github.com/Wenyueh/LLM-RecSys-ID/issues/2).

# Log files
All experiment log files are under the log directory. 

The log files can be used as references for replicating experiments.

You can use the log files to check the hyperparameter settings.

# Random Indexing (RID)
```
 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
   --master_port 123227 \
   main.py \
      --distributed --multiGPU \
      --task beauty \
         --seed 2022 \
         --warmup_prop 0.05 \
         --lr 1e-3 \
         --clip 1.0 \
         --model_type 't5-small' \
         --epochs 20 \
         --gpu '0,1' \
         --logging_step 1000 \
         --logging_dir 'log/pretrain_t5_small_beauty_random.log' \
         --model_dir 'model/pretrain_t5_small_beauty_random.pt' \
         --train_sequential_item_batch 64 \
         --whole_word_embedding shijie \
         --item_representation random_number \
         --data_order random \
         --random_initialization_embedding \
         --min_random_number 1000 \
         --max_random_number 13000
 ```
 
 # Independent Indexing (IID)
 ```
 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
   --master_port 123227 \
   main.py \
      --distributed --multiGPU \
      --task beauty \
         --seed 2022 \
         --warmup_prop 0.05 \
         --lr 1e-3 \
         --clip 1.0 \
         --model_type 't5-small' \
         --epochs 20 \
         --gpu '0,1' \
         --logging_step 1000 \
         --logging_dir 'log/pretrain_t5_small_beauty_independent.log' \
         --model_dir 'model/pretrain_t5_small_beauty_independent.pt' \
         --train_sequential_item_batch 64 \
         --whole_word_embedding shijie \
         --item_representation no_tokenization \
         --data_order random
 ```
 
 # Title Indexing (TID)
 ```
 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
   --master_port 123227 \
   main.py \
      --distributed --multiGPU \
      --task beauty \
         --seed 2022 \
         --warmup_prop 0.05 \
         --lr 1e-3 \
         --clip 1.0 \
         --model_type 't5-small' \
         --epochs 20 \
         --gpu '0,1' \
         --logging_step 1000 \
         --logging_dir 'log/pretrain_t5_small_beauty_title.log' \
         --model_dir 'model/pretrain_t5_small_beauty_title.pt' \
         --train_sequential_item_batch 64 \
         --whole_word_embedding shijie \
         --item_representation title \
         --data_order random
 ```
 
 # Sequential Indexing (SID, time sensitive)
 ```
 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
   --master_port 123227 \
   main.py \
      --distributed --multiGPU \
      --task beauty \
         --seed 2022 \
         --warmup_prop 0.05 \
         --lr 1e-3 \
         --clip 1.0 \
         --model_type 't5-small' \
         --epochs 20 \
         --gpu '0,1' \
         --logging_step 1000 \
         --logging_dir 'log/pretrain_t5_small_beauty_sequential_time_sensitive.log' \
         --model_dir 'model/pretrain_t5_small_beauty_title.pt' \
         --train_sequential_item_batch 64 \
         --whole_word_embedding shijie \
         --item_representation None \
         --data_order remapped_sequential \
         --remapped_data_order original
 ```
 
 # Collaborative Indexing (CID)
 
 need to run CID_generation.py to generate files, which requires the input file of remapped_sequential_data.txt
 
 ```
 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
   --master_port 123227 \
   main.py \
      --distributed --multiGPU \
      --task beauty \
         --seed 2022 \
         --warmup_prop 0.05 \
         --lr 1e-3 \
         --clip 1.0 \
         --model_type 't5-small' \
         --epochs 20 \
         --gpu '0,1' \
         --logging_step 1000 \
         --logging_dir 'log/pretrain_t5_small_beauty_CF.log' \
         --model_dir 'model/pretrain_t5_small_beauty_CF.pt' \
         --train_sequential_item_batch 64 \
         --whole_word_embedding shijie \
         --item_representation CF \
         --data_order remapped_sequential \
         --remapped_data_order original \
         --cluster_size 500 \
         --cluster_number 20
 ```

# Semantic Indexing (SemID)
 ```
 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
   --master_port 123227 \
   main.py \
      --distributed --multiGPU \
      --task beauty \
         --seed 2022 \
         --warmup_prop 0.05 \
         --lr 1e-3 \
         --clip 1.0 \
         --model_type 't5-small' \
         --epochs 20 \
         --gpu '0,1' \
         --logging_step 1000 \
         --logging_dir 'log/pretrain_t5_small_beauty_semantics.log' \
         --model_dir 'model/pretrain_t5_small_beauty_semantics.pt' \
         --train_sequential_item_batch 64 \
         --whole_word_embedding shijie \
         --item_representation content_based \
         --data_order random
 ```
 
 # Hybrid Indexing (HID: CID+IID)
 ```
 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
   --master_port 123227 \
   main.py \
      --distributed --multiGPU \
      --task beauty \
         --seed 2022 \
         --warmup_prop 0.05 \
         --lr 1e-3 \
         --clip 1.0 \
         --model_type 't5-small' \
         --epochs 20 \
         --gpu '0,1' \
         --logging_step 1000 \
         --logging_dir 'log/pretrain_t5_small_beauty_CID+IID.log' \
         --model_dir 'model/pretrain_t5_small_beauty_CID_IID.pt' \
         --train_sequential_item_batch 64 \
         --whole_word_embedding shijie \
         --item_representation CF \
         --data_order remapped_sequential \
         --cluster_size 500 \
         --cluster_number 20 \
         --last_token_no_repetition
 ```

# Citation
Please cite the following papers corresponding to the repository:
```
@article{hua2023index,
  title={How to Index Item IDs for Recommendation Foundation Models},
  author={Hua, Wenyue and Xu, Shuyuan and Ge, Yingqiang and Zhang, Yongfeng},
  journal={SIGIR-AP},
  year={2023}
}
@article{xu2023openp5,
  title={OpenP5: Benchmarking Foundation Models for Recommendation},
  author={Shuyuan Xu and Wenyue Hua and Yongfeng Zhang},
  journal={arXiv:2306.11134},
  year={2023}
}
```
