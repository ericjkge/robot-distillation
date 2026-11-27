# shellcheck shell=bash disable=SC2034
# Sourced after a model file: any CUDA GPU that PEAR's H100 and GB300 recipes do not cover. The model runs on vLLM's own
# Qwen3.5 implementation and verl's FSDP path with FlashAttention-2, without PEAR's Hopper/Blackwell kernels or the
# LMCache sidecar. HARDWARE_ARGS replace the model file's arguments of the same keys.
# the machine's GPUs (NODE_GPUS when the machine running this is not one of the cluster's); the actor's Ulysses sequence
# parallelism spans the trainer's, as in PEAR's recipes
GPUS_PER_NODE=${NODE_GPUS:-$(nvidia-smi -L | wc -l)}
SP=${TRAINER_GPUS:-$GPUS_PER_NODE}
TOK_PER_GPU=$(((MAX_MODEL_LEN + SP - 1) / SP))
HARDWARE_ARGS=(
  actor_rollout_ref.actor.fsdp_config.use_gpu_backend=false
  actor_rollout_ref.rollout.kernels=null
  "+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2"
  "+actor_rollout_ref.rollout.custom.lmcache={enable: false}"
  actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size="$SP"
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$TOK_PER_GPU"
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$TOK_PER_GPU"
  # the actor's fp32 state on the CPU (FSDP2's offload_policy), as SFT keeps it: PEAR's optimizer_offload brings it onto
  # the GPUs for Adam's step, which a 48 GB GPU cannot hold; verl allows optimizer_offload only without offload_policy
  actor_rollout_ref.actor.fsdp_config.offload_policy=true
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
  actor_rollout_ref.actor.fsdp_config.optimizer_offload_step=false
)
# the distillation teacher's vLLM on 48 GB GPUs (TP 2: 26 GiB of weights each, with the fp32 head and vision tower):
# verl's default memory share (0.5) leaves no KV cache, and its prompt log-probs, which vLLM computes outside its
# budget (fp32 logits over the vocabulary per prefill chunk), need chunks of 2,048 tokens rather than 8,192
TEACHER_HARDWARE_ARGS=(
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.9
  distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=2048
)
