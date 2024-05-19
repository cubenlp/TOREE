# All can be done using a single 3090 GPU

# modify the following args according to your own environment:
# model_name_or_path: the path to the pre-trained model
# dataset: the path of TOREE test set
# template: the template used for specific LLMs
# output_dir: the path to save the generated responses
# per_device_eval_batch_size: the batch size for evaluation
# max_samples: the maximum number of samples to be evaluated



# Baichuan2
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path /sshfs/pretrains/baichuan-inc/Baichuan2-7B-Chat \
    --dataset TOREE_test_2shot \
    --template default \
    --flash_attn False \
    --shift_attn False \
    --finetuning_type lora \
    --output_dir TOREE/results/vanilla_2shot \
    --per_device_eval_batch_size 8 \
    --max_samples 10000 \
    --predict_with_generate \
    --top_p 0.95 \
    --temperature 0.95 \
    --cutoff_len 1024 \
    --fp16

# chinese-alpaca2
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path /sshfs/pretrains/hfl/chinese-alpaca-2-7b \
    --dataset TOREE_test_2shot \
    --template default \
    --flash_attn False \
    --shift_attn False \
    --finetuning_type lora \
    --output_dir TOREE/chinese-alpaca2-7b/results/vanilla_2shot \
    --per_device_eval_batch_size 2 \
    --max_samples 10000 \
    --predict_with_generate \
    --top_p 0.95 \
    --temperature 0.95 \
    --cutoff_len 1024 \
    --fp16

# chatglm3-6b
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path /sshfs/pretrains/THUDM/chatglm3-6b \
    --dataset TOREE_test_2shot \
    --template default \
    --flash_attn False \
    --shift_attn False \
    --finetuning_type lora \
    --output_dir TOREE/chatglm3-6b/results/vanilla_2shot \
    --per_device_eval_batch_size 8 \
    --max_samples 10000 \
    --predict_with_generate \
    --top_p 0.95 \
    --temperature 0.95 \
    --cutoff_len 1024 \
    --fp16

# Qwen-7b
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path /sshfs/pretrains/Qwen/Qwen-7B-Chat \
    --dataset TOREE_test_2shot \
    --template default \
    --flash_attn False \
    --shift_attn False \
    --finetuning_type lora \
    --output_dir TOREE/Qwen-7b/results/vanilla_2shot \
    --per_device_eval_batch_size 8 \
    --max_samples 10000 \
    --predict_with_generate \
    --top_p 0.95 \
    --temperature 0.95 \
    --cutoff_len 1024 \
    --fp16