# !/bin/bash
echo " Running GTA IM Training EXP"

#Full time --pred_seq_len 10
echo "Full time start2"


CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0 --l_cos 0.1  --tag 2  && echo "EXP 3 Launched, with learn adj,cos loss 0.1" &
P0=$!

CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0  --tag 2  && echo "EXP 4 Launched, with learn adj,norm loss 0.1" &
P1=$!

CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0.1  --tag 2  && echo "EXP 5 Launched, with learn adj,norm loss 0.1,cos loss 0.1" &
P2=$!




CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0 --l_cos 0.1  --tag 3  && echo "EXP 3 Launched, with learn adj,cos loss 0.1" &
P3=$!

CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0  --tag 3  && echo "EXP 4 Launched, with learn adj,norm loss 0.1" &
P4=$!

CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0.1  --tag 3  && echo "EXP 5 Launched, with learn adj,norm loss 0.1,cos loss 0.1" &
P5=$!


wait $P0 $P1 $P2 $P3 $P4 $P5