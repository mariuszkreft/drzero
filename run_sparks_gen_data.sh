# Dr. Zero — Datensynthese mit dem trainierten Proposer, Spark-Variante.
# Ableitung von iter1_gen_data.sh.
# Env: DRZERO_CORPUS, DRZERO_MODEL, DRZERO_GPUS, DRZERO_STEP, DRZERO_PARTITION
set -x

gpus=${DRZERO_GPUS:-4}
tp=1
dp=$gpus
sample_size=5
rollout_memory_utilization=0.75
model=${DRZERO_MODEL:-Qwen/Qwen2.5-3B-Instruct}
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
corpus=${DRZERO_CORPUS:-./corpus}

hop_ratio=${1:-4321}
if [ $# -ge 1 ]; then shift; fi

algorithm=grpo_batch
grpo_group_size=1
reward_group_size=5

challenger_step=${DRZERO_STEP:-50}
data_partition=${DRZERO_PARTITION:-1}

EXP_DIR="checkpoints/dr-zero"
MODEL_PATH="challenger_iter1_ratio${hop_ratio}_${algorithm}_group${grpo_group_size}-${reward_group_size}_${model_name}"
CKPT_PATH="${EXP_DIR}/${MODEL_PATH}/global_step_${challenger_step}"

TOOL_CONFIG="./config/search_tool_config.yaml"
TRAIN_DATA="./data/zero_ratio${hop_ratio}.parquet"
TRAIN_DATA_OUT="./data/zero_${MODEL_PATH}.parquet"

if [ -z "$DRZERO_RETRIEVAL_URL" ]; then
  python search/retrieval_server.py \
      --index_path="$corpus/e5_Flat.index" \
      --corpus_path="$corpus/wiki-18.jsonl" \
      --retriever_model='intfloat/e5-base-v2' \
      --retriever_name='e5' --faiss_gpu --topk 3 &
  sleep 20
fi

python -m verl.trainer.main_generation \
    --config-path="./config" --config-name='search_multiturn_grpo' \
    +ckpt_path=$CKPT_PATH \
    data.prompt_key=prompt +data.path=$TRAIN_DATA +data.partition=$data_partition \
    +data.batch_size=256 +data.output_path=$TRAIN_DATA_OUT \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=$sample_size \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.prompt_length=1536 \
    actor_rollout_ref.rollout.response_length=2560 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_memory_utilization \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tp \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG \
    trainer.n_gpus_per_node=$gpus trainer.nnodes=1 $@