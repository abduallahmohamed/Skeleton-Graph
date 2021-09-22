# !/bin/bash
echo " Running PROX Training EXP"

#Full time --pred_seq_len 10
echo "Full time start"
# CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.03 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --learn_A --torso_joint 13 --eval_only && echo "EXP 0 Launched, standard model" &
# P0=$!
# wait $P0

# CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.03 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450  --torso_joint 13 --eval_only  && echo "EXP 1 Launched, with learn adj" &
# P0=$!
# wait $P0


# CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.03 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.005 --l_cos 0.005   --torso_joint 13 --eval_only  && echo "EXP 2 Launched, with learn adj,cons loss 0.01" &
# P0=$!
# wait $P0


# CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.03 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0 --l_cos 0.005   --torso_joint 13 --eval_only  && echo "EXP 3 Launched, with learn adj,cos loss 0.01" &
# P0=$!
# wait $P0

# CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.03 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.005 --l_cos 0   --torso_joint 13 --eval_only && echo "EXP 4 Launched, with learn adj,norm loss 0.01" &
# P0=$!
# wait $P0
# echo "Full time ends"

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.03 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0.1 --background_back  --tag 1 --torso_joint 13 --eval_only  && echo "EXP 6 vision image" &
P0=$!
wait $P0

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.03 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.1 --l_cos 0.1 --video_back --tag 1  --torso_joint 13 --eval_only  && echo "EXP 7 vision video" &
P3=$
wait $P0



