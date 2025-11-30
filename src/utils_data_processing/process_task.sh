set -e


#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_bird_with_evidence \
#    --mode t2s_fullS-d-wT \
#    --dataset_path ../data/bird_train_full.json \
#    --target_dataset_path ../../LLMsFT_text2sql/data/sft_t2s_fullS-d-wT_bird_train.json


#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_bird_with_evidence \
#    --mode sg-Dsl-fullS-d-wT \
#    --dataset_path ../data/bird_train_full.json \
#    --target_dataset_path ../../LLMsFT_text2sql/data/sft_sg-Dsl-fullS-d-wT_bird_train.json
#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_spider \
#    --mode sg-Dsl-fullS-d-wT \
#    --dataset_path ../data/spider_train_full.json.json \
#    --target_dataset_path ../../LLMsFT_text2sql/data/sft_sg-Dsl-fullS-d-wT_spider_train.json
# maxL 4098
#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_bird_with_evidence \
#    --mode sg-t2sTsl-fullS-d-wT \
#    --dataset_path ../data/bird_train_full.json \
#    --target_dataset_path ../../LLMsFT_text2sql/data/sft_sg-t2sTsl-fullS-d-wT_bird_train.json
#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_spider \
#    --mode sg-t2sTsl-fullS-d-wT \
#    --dataset_path ../data/spider_train_full.json.json \
#    --target_dataset_path ../../LLMsFT_text2sql/data/sft_sg-t2sTsl-fullS-d-wT_spider_train.json

#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_bird_with_evidence \
#    --mode sg_t2s-sl_briefS \
#    --dataset_path ../data/bird_train_full.json \
#    --target_dataset_path ../../LLMsFT_text2sql/data/sft_sg_t2s-sl_briefS_bird_train.json

#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_bird_with_evidence \
#    --mode t2s_idealS-d \
#    --dataset_path ../data/bird_train_full.json \
#    --target_dataset_path ../../LLMsFT_text2sql/data/sft_t2s_idealS-d_bird_train.json

# SL-not in codes style (No-type)
#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_bird_with_evidence \
#    --mode sg_t2s-sl_fullS-d \
#    --dataset_path ../data/bird_train_full.json \
#    --target_dataset_path ../../LLMsFT_text2sql/data/sft_sg_t2s-sl_fullS-d_bird_train.json

CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
    --sic_path ../sic_ckpts/sic_bird_with_evidence \
    --mode sg-t2sTsl-fullS-d \
    --dataset_path ../data/bird_train_full.json \
    --target_dataset_path ../../LLMsFT_text2sql/data/sft_sg-t2sTsl-fullS-d_bird_train.json
#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_spider \
#    --mode sg-t2sTsl-fullS-d \
#    --dataset_path ../data/spider_train_full.json.json \
#    --target_dataset_path ../../LLMsFT_text2sql/data/sft_sg-t2sTsl-fullS-d_spider_train.json
# SL-not in codes style (No-type) 4656
#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_spider \
#    --mode sg-t2sTsl-fullS-d \
#    --dataset_path ../data/spider_train_full.json.json \
#    --target_dataset_path ../../LLMsFT_text2sql/data/sft_sg-t2sTsl-fullS-d_spider_train.json
#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_spider \
#    --mode sg-t2sTsl-fullS-d \
#    --dataset_path ../data/spider_dev_full.json
#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_bird_with_evidence \
#    --mode sg-t2sTsl-fullS-d \
#    --dataset_path ../data/bird_dev_full.json

#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_bird_with_evidence \
#    --mode sg-t2sTsl-fullS-d \
#    --dataset_path ../data/bird_dev_full.json \
#    --target_dataset_path ../../LLMsFT_text2sql/data/sft_sg-t2sTsl-fullS-d_bird_dev.json

## 0616-8:00pm BIRD SL in codes style maxL 12179
#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_bird_with_evidence \
#    --mode sg-t2sTsl-fullS-d-codesStyle \
#    --dataset_path ../data/bird_train_full.json \
#    --target_dataset_path ../../LLMsFT_text2sql/data/sft_sg-t2sTsl-fullS-d-codesStyle_bird_train.json
## 0616-8:00pm SPIDER SL in codes style maxL 6539
#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_spider \
#    --mode sg-t2sTsl-fullS-d-codesStyle \
#    --dataset_path ../data/spider_train_full.json.json \
#    --target_dataset_path ../../LLMsFT_text2sql/data/sft_sg-t2sTsl-fullS-d-codesStyle_spider_train.json
# maxL 4727
#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_bird_with_evidence \
#    --mode sg-t2sTsl-fullS-d-codesStyle \
#    --dataset_path ../data/bird_dev_full.json
## maxL 1828
#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_spider \
#    --mode sg-t2sTsl-fullS-d-codesStyle \
#    --dataset_path ../data/spider_dev_full.json

#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_bird_with_evidence \
#    --mode t2s_fullS-d \
#    --dataset_path ../data/bird_train_full.json \
#    --target_dataset_path ../../LLMsFT_text2sql/data/sft_t2s_fullS-d_bird_train.json


##  BIRD T2S in codes style
#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_bird_with_evidence \
#    --mode t2s-codesStyle-json \
#    --dataset_path ../data/bird_train_full.json \
#    --target_dataset_path ../../LLMsFT_text2sql/data/sft_t2s-codesStyle-json_bird_train.json

##  SPIDER T2S in codes style maxL 1457
#CUDA_VISIBLE_DEVICES=0 python -u process_task.py \
#    --sic_path ../sic_ckpts/sic_spider \
#    --mode t2s-codesStyle-json \
#    --dataset_path ../data/spider_train_full.json.json \
#    --target_dataset_path ../../LLMsFT_text2sql/data/sft_t2s-codesStyle-json_spider_train.json


#CUDA_VISIBLE_DEVICES=7 nohup python -u process_task_inserver.py \
#    --sic_path sic_ckpts/sic_bird_with_evidence \
#    --mode t2s-codesStyle-FAR-json \
#    --schema_linking_file Results_evalTrain_sg_t2s-sl_fullS-d_codellama13b_beam-2_checkpoint-1800_bird_train_relevant_columns.json \
#    --dataset_path data/bird_train_full.json \
#    --target_dataset_path data/sft_t2s-codesStyle-FAR-json_bird_train.json > log_process_task_inserver_t2s-codesStyle-FAR-json.log 2>&1 &
#

