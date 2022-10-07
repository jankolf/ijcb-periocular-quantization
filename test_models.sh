#!/bin/bash
export OMP_NUM_THREADS=2
python test_models_quantized.py --gpu="4" --model_folder="mobilefacenet_identification_flip_quantized_wq8-6_aq8-6" --protocol="closed_world" --flip_images 2>&1 | tee "ident_q.txt"
python test_models_quantized.py --gpu="4" --model_folder="mobilefacenet_verification_flip_quantized_wq8-6_aq8-6" --protocol="open_world" --flip_images 2>&1 | tee "ver_q.txt"
