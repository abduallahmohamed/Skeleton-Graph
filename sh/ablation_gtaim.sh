# !/bin/bash
echo " Running GTA IM Arch Ablation EXP"






CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 3 --n_txpcnn 1  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 3,1" &
P0=$!
wait $P0

CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 3 --n_txpcnn 3  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 3,3" &
P0=$!
wait $P0

CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 3 --n_txpcnn 5  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 3,5" &
P0=$!
wait $P0

CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 3 --n_txpcnn 7  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 3,7" &
P0=$!
wait $P0



CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 1  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 5,1" &
P0=$!
wait $P0

CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 3  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 5,3" &
P0=$!
wait $P0

CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 5  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 5,5" &
P0=$!
wait $P0

CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 7  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 5,7" &
P0=$!
wait $P0




CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 7 --n_txpcnn 1  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 7,1" &
P0=$!
wait $P0

CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 7 --n_txpcnn 3  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 7,3" &
P0=$!
wait $P0

CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 7 --n_txpcnn 5  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 7,5" &
P0=$!
wait $P0

CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 7 --n_txpcnn 7  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 7,7" &
P0=$!
wait $P0


CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 1  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 1,1" &
P0=$!
wait $P0


CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 3  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 1,3" &
P0=$!
wait $P0


CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 1,5" &
P0=$!
wait $P0


CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 7  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 1,7" &
P0=$!
wait $P0


CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 9  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 1,9" &
P0=$!
wait $P0


CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 3 --n_txpcnn 9  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 3,9" &
P0=$!
wait $P0


CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 9  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 5,9" &
P0=$!
wait $P0


CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 7 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 7,11" &
P0=$!
wait $P0

CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 1,11" &
P0=$!
wait $P0


CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 3 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 3,11" &
P0=$!
wait $P0


CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 5,11" &
P0=$!
wait $P0


CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.01 --n_stgcnn 7 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450   && echo "EXP 0 Launched, 7,11" &
P0=$!
wait $P0
