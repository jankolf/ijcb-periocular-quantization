#!/bin/bash
export OMP_NUM_THREADS=3

for i in `find ./configs/ -type f -name "run_normal_*.pkl" | cat | sort`;
do
  echo "$i"
  output_file="${i}.txt"
  CUDA_VISIBLE_DEVICES=4,5,6 python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 \
    --node_rank=0 --master_addr="127.0.0.1" --master_port=26001 distributed_train_normal.py $i 2>&1 | tee $output_file

  
done

for i in `find ./configs/ -type f -name "run_quantized_*.pkl" | cat | sort`;
do
  echo "$i"
  output_file="${i}.txt"
  CUDA_VISIBLE_DEVICES=4,5,6 python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 \
    --node_rank=0 --master_addr="127.0.0.1" --master_port=26001 distributed_train_quantization.py $i 2>&1 | tee $output_file

  
done

python test_models_normal.py --gpu="4" --model_folder="models" --protocol="closed_world" --flip_images
python test_models_quantized.py --gpu="4" --model_folder="models_quantized" --protocol="closed_world" --flip_images
