# !/bin/bash
echo " Running PROX Training EXP"



CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0.1 --background_back  --tag 1  && echo "EXP 6 vision image" &
P0=$!

CUDA_VISIBLE_DEVICES=4 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0.1 --background_back  --tag 2  && echo "EXP 6 vision image" &
P1=$!

CUDA_VISIBLE_DEVICES=5 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0.1 --background_back  --tag 3  && echo "EXP 6 vision image" &
P2=$!

echo "Video"

CUDA_VISIBLE_DEVICES=6 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0.1 --video_back  --tag 1  && echo "EXP 7 vision video" &
P3=$!

CUDA_VISIBLE_DEVICES=7 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0.1 --video_back  --tag 2  && echo "EXP 7 vision video" &
P4=$!

CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0.1 --video_back  --tag 3  && echo "EXP 7 vision video" &
P5=$!


wait $P0 $P1 $P2 $P3 $P4 $P5


