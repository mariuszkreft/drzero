# Dr. Zero — Proposer-Lauf, Variante fuer den Spark-Cluster (arm64, unified memory).
# Ableitung von iter1_challenger.sh: gleiche Argumente, kleinere Batches, 4 statt 8 Karten.
# Env: DRZERO_CORPUS, DRZERO_MODEL, DRZERO_RETRIEVAL_URL (externer Retriever), DRZERO_GPUS
set -x

gpus=${DRZERO_GPUS:-4}
tp=2
dp=$((gpus / tp))
batch_per_gpu=1
rollout_memory_utilization=0.2
model=${DRZERO_MODEL:-Qwen/Qwen2.5-3B-Instruct}
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
corpus=${DRZERO_CORPUS:-./corpus}

hop_ratio=${1:-4321}
if [ $# -ge 1 ]; then shift; fi

algorithm=grpo_batch
grpo_group_size=1
reward_group_size=5

TOOL_CONFIG="./config/search_tool_config.yaml"
TRAIN_DATA="./data/zero_ratio${hop_ratio}.parquet"
VAL_DATA="./data/test.parquet"

if [ -z "$DRZERO_RETRIEVAL_URL" ]; then
  python search/retrieval_server.py \
      --index_path="$corpus/e5_Flat.index" \
      --corpus_path="$corpus/wiki-18.jsonl" \
      --retriever_model='intfloat/e5-base-v2' \
      --retriever_name='e5' --faiss_gpu --topk 3 &
  sleep 20
fi

python -m sglang.launch_server \
    --model=$model --port=8001 --tool-call-parser=qwen25 \
    --mem-fraction-static=$rollout_memory_utilization \
    --dp-size=$dp --tp-size=$tp --log-level=error &

sleep 40

python -m verl.trainer.main_ppo \
    --config-path="./config" --config-name='search_multiturn_grpo' \
    data.train_files=$TRAIN_DATA data.val_files=$VAL_DATA \
    data.train_batch_size=128 \
    algorithm.use_kl_in_reward=False algorithm.adv_estimator=$algorithm \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.actor.grad_clip=0.1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$batch_per_gpu \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=$grpo_group_size \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_memory_utilization \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tp \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$batch_per_gpu \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$batch_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG \
    reward_model.reward_manager=batch \
    custom_reward_function.name=compute_challenger_score_batch \
    custom_reward_function.path=verl/custom_reward/reward_function.py \
    custom_reward_function.reward_kwargs.model_name=$model \
    custom_reward_function.reward_kwargs.base_url="http://127.0.0.1:8001" \
    custom_reward_function.reward_kwargs.reward_rollout_n=$reward_group_size \
    trainer.logger='["console"]' \
    trainer.project_name="dr-zero" \
    trainer.experiment_name="challenger_iter1_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}-${reward_group_size}_${model_name}" \
    trainer.n_gpus_per_node=$gpus trainer.nnodes=1 \
    trainer.save_freq=25 trainer.test_freq=-1 trainer.val_before_train=False \
    trainer.total_epochs=1 $@