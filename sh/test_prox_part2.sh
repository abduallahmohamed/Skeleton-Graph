# !/bin/bash
echo " Running PROX Test EXP"

#Full time --pred_seq_len 10

echo "Group 1"

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0 --l_cos 0.1  --tag 4    --torso_joint 13 --eval_only  && echo "EXP 3 Launched, with learn adj,cos loss 0.1" &
P0=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0 --l_cos 0.1  --tag 5    --torso_joint 13 --eval_only  && echo "EXP 3 Launched, with learn adj,cos loss 0.1" &
P1=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0 --l_cos 0.1  --tag 6    --torso_joint 13 --eval_only  && echo "EXP 3 Launched, with learn adj,cos loss 0.1" &
P2=$!
wait $P0 $P1 $P2



echo "Group 2"


CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0  --tag 4    --torso_joint 13 --eval_only  && echo "EXP 3 Launched, with learn adj,cos loss 0.1" &
P0=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0  --tag 5    --torso_joint 13 --eval_only  && echo "EXP 3 Launched, with learn adj,cos loss 0.1" &
P1=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0  --tag 6    --torso_joint 13 --eval_only  && echo "EXP 3 Launched, with learn adj,cos loss 0.1" &
P2=$!
wait $P0 $P1 $P2

echo "Group 3"



CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0.1  --tag 4    --torso_joint 13 --eval_only  && echo "EXP 3 Launched, with learn adj,cos loss 0.1" &
P0=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0.1  --tag 6    --torso_joint 13 --eval_only  && echo "EXP 3 Launched, with learn adj,cos loss 0.1" &
P1=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0.1  --tag 6    --torso_joint 13 --eval_only  && echo "EXP 3 Launched, with learn adj,cos loss 0.1" &
P2=$!
wait $P0 $P1 $P2