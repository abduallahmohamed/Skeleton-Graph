
# !/bin/bash
echo " Running GTA IM Test EXP"

#Full time --pred_seq_len 10


CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0 --l_cos 0.1  --tag 1   --torso_joint 13 --eval_only  && echo "EXP 3 Launched, with learn adj,cos loss 0.1" &
P0=$!
CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0 --l_cos 0.1  --tag 2    --torso_joint 13 --eval_only  && echo "EXP 3 Launched, with learn adj,cos loss 0.1" &
P1=$!
CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0 --l_cos 0.1  --tag 3    --torso_joint 13 --eval_only  && echo "EXP 3 Launched, with learn adj,cos loss 0.1" &
P2=$!
echo " section 1"
wait $P0 $P1 $P2

CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0  --tag 1    --torso_joint 13 --eval_only  && echo "EXP 4 Launched, with learn adj,norm loss 0.1" &
P3=$!
CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0  --tag 2    --torso_joint 13 --eval_only  && echo "EXP 4 Launched, with learn adj,norm loss 0.1" &
P4=$!
CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0  --tag 3    --torso_joint 13 --eval_only  && echo "EXP 4 Launched, with learn adj,norm loss 0.1" &
P5=$!
echo " section 2"
wait $P3 $P4 $P5

CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0.1  --tag 1    --torso_joint 13 --eval_only  && echo "EXP 5 Launched, with learn adj,norm loss 0.1,cos loss 0.1" &
P6=$!
CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0.1  --tag 2    --torso_joint 13 --eval_only  && echo "EXP 5 Launched, with learn adj,norm loss 0.1,cos loss 0.1" &
P7=$!
CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0.1  --tag 3    --torso_joint 13 --eval_only  && echo "EXP 5 Launched, with learn adj,norm loss 0.1,cos loss 0.1" &
P8=$!
echo " section 3"


wait  $P6 $P7 $P8