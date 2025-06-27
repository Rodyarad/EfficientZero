set -ex
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0
export WANDB_ENTITY="r-rodionvahitoff"

python main.py --env BreakoutNoFrameskip-v4 --case atari_test --opr train --force \
  --num_gpus 1 --num_cpus 10 --cpu_actor 5 --gpu_actor 5 \
  --seed 0 \
  --p_mcts_num 4 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info 'EfficientZero-V1' \
  --object_store_memory 2147483648