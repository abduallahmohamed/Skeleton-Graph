# !/bin/bash
echo " Running GTA IM Test EXP"

#Full time --pred_seq_len 10
echo "Full time start"
CUDA_VISIBLE_DEVICES=1 python3 visualization.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450  --tag 1   --torso_joint 13 --eval_only --flag 0

CUDA_VISIBLE_DEVICES=2 python3 visualization.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0.1  --tag 2    --torso_joint 13 --eval_only  --flag 1