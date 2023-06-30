# How to Index Item IDs for Recommendation Foundation Models

# Abstract

Recommendation foundation models utilize large language models as their backbone, converting recommendation tasks into natural language tasks. Items are represented as texts and tokenized by language model tokenizers, resulting in multiple tokens and corresponding embeddings for each item, in contrast to the single vector representations in traditional recommender systems. Therefore, creating item ID indexing compatible with language models is essential for recommendation foundation models. In this study, we systematically examine the item indexing problem for recommendation foundation models, using P5 as the representative backbone model and replicating its results with various indexing methods. To emphasize the importance of item indexing, we discuss the issues of several basic methods, such as independent indexing, title indexing, and random indexing, from empirical perspectives. We then propose four simple yet effective solutions, including sequential indexing, collaborative indexing, semantic (content-based) indexing, and hybrid indexing. Our reproducibility study of P5 highlights the significant influence of indexing methods on the model performance, and our results on real-world datasets validate the effectiveness of our proposed solutions.

<img width="911" alt="Screen Shot 2023-04-24 at 1 58 10 PM" src="https://user-images.githubusercontent.com/28013619/234078088-3d020437-cf7b-4063-bf8b-940d8fa44dd6.png">

# Before running script
download meta data from here: https://drive.google.com/file/d/1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G/view, put in the data/ directory

build environment:
```
conda create -n itemrep -y python=3.9.7 && conda activate itemrep
pip install -r requirements.txt
```

# random indexing
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
 
 # independent indexing
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
 
 # title indexing
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
 
 # sequential indexing (time sensitive)
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
 
 # collaborative indexing 
 
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

# semantic indexing
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
 
 # hybrid indexing (CID+IID)
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
  journal={arXiv:2305.06569},
  year={2023}
}
```
