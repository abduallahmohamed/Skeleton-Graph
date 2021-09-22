# !/bin/bash
echo " Running GTA IM Test EXP"



echo "Standard"
CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --learn_A --tag 1   --torso_joint 12 --eval_only  && echo "EXP 0 Launched, standard model" &
P0=$!

CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --learn_A --tag 2   --torso_joint 12 --eval_only  && echo "EXP 0 Launched, standard model" &
P1=$!


CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --learn_A --tag 3   --torso_joint 12 --eval_only  && echo "EXP 0 Launched, standard model" &
P2=$!


CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --learn_A --tag 4   --torso_joint 12 --eval_only  && echo "EXP 0 Launched, standard model" &
P3=$!


CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --learn_A --tag 5   --torso_joint 12 --eval_only  && echo "EXP 0 Launched, standard model" &
P4=$!

wait $P0 $P1 $P2 $P3 $P4
echo "Standard Ends"



echo "Learn Adj"
CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450  --tag 1   --torso_joint 12 --eval_only  && echo "EXP 1 Launched, standard model" &
P0=$!

CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450  --tag 2   --torso_joint 12 --eval_only  && echo "EXP 1 Launched, standard model" &
P1=$!


CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450  --tag 3   --torso_joint 12 --eval_only  && echo "EXP 1 Launched, standard model" &
P2=$!


CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450  --tag 4   --torso_joint 12 --eval_only  && echo "EXP 1 Launched, standard model" &
P3=$!


CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450  --tag 5   --torso_joint 12 --eval_only  && echo "EXP 1 Launched, standard model" &
P4=$!

wait $P0 $P1 $P2 $P3 $P4
echo "Learn Adj Ends"

