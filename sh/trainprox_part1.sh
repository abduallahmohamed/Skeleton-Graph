# !/bin/bash
echo " Running PROX Training EXP"



echo "Standard"
CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --learn_A --tag 1  && echo "EXP 0 Launched, standard model" &
P0=$!

CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --learn_A --tag 2  && echo "EXP 0 Launched, standard model" &
P1=$!


CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --learn_A --tag 3  && echo "EXP 0 Launched, standard model" &
P2=$!


CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --learn_A --tag 4  && echo "EXP 0 Launched, standard model" &
P3=$!


CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --learn_A --tag 5  && echo "EXP 0 Launched, standard model" &
P4=$!

wait $P0 $P1 $P2 $P3 $P4
echo "Standard Ends"



echo "Learn Adj"
CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450  --tag 1  && echo "EXP 1 Launched, Learn Adj model" &
P0=$!

CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450  --tag 2  && echo "EXP 1 Launched, Learn Adj model" &
P1=$!


CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450  --tag 3  && echo "EXP 1 Launched, Learn Adj model" &
P2=$!


CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450  --tag 4  && echo "EXP 1 Launched, Learn Adj model" &
P3=$!


CUDA_VISIBLE_DEVICES=1 python3 train.py --lr 0.05 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450  --tag 5  && echo "EXP 1 Launched, Learn Adj model" &
P4=$!

wait $P0 $P1 $P2 $P3 $P4
echo "Learn Adj Ends"

