set -x
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0
MODEL_PATH=/mnt/home/LLaMA-Factory/saves/Qwen2.5-7B-Instruct-Ret/full/sft

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=sheryc/DROP_processed@train \
    data.val_files=sheryc/DROP_processed@validation \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.format_prompt=./examples/format_prompt/retrieve_format.jinja \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.score_function=./examples/score_function/retrieve.py:compute_score \
    trainer.experiment_name=qwen2_5_7b_retrieval \
    trainer.n_gpus_per_node=8
