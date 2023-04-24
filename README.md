# Item Indexing Methods for Recommendation Foundation Models: A Reproducibility Study

<img width="911" alt="Screen Shot 2023-04-24 at 1 58 10 PM" src="https://user-images.githubusercontent.com/28013619/234078088-3d020437-cf7b-4063-bf8b-940d8fa44dd6.png">

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
 
