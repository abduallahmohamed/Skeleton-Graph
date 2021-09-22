# !/bin/bash
echo " Running GTA IM Test EXP"

#Full time --pred_seq_len 10
echo "Full time start"
CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --learn_A  --torso_joint 13 --eval_only  && echo "EXP 0 Launched, standard model" &
P0=$!
wait $P0

CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450  --torso_joint 13 --eval_only  && echo "EXP 1 Launched, with learn adj" &
P0=$!
wait $P0


CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0.01 --l_cos 0.01 --torso_joint 13 --eval_only  && echo "EXP 2 Launched, with learn adj,cons loss 0.01" &
P0=$!
wait $P0


CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0 --l_cos 0.01 --torso_joint 13 --eval_only  && echo "EXP 3 Launched, with learn adj,cos loss 0.01" &
P0=$!
wait $P0

CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --use_cons --l_norm 0.01 --l_cos 0 --torso_joint 13 --eval_only  && echo "EXP 4 Launched, with learn adj,norm loss 0.01" &
P0=$!
wait $P0
echo "Full time ends"

# #0.25 time --pred_seq_len 2
# echo "0.25 time start"
# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --learn_A --pred_seq_len 2 --torso_joint 13 --eval_only  && echo "EXP 0 Launched, standard model" &
# P0=$!
# wait $P0

# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --pred_seq_len 2 --torso_joint 13 --eval_only  && echo "EXP 1 Launched, with learn adj" &
# P0=$!
# wait $P0


# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --pred_seq_len 2 --use_cons --l_norm 0.01 --l_cos 0.01 --torso_joint 13 --eval_only  && echo "EXP 2 Launched, with learn adj,cons loss 0.01" &
# P0=$!
# wait $P0


# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --pred_seq_len 2 --use_cons --l_norm 0 --l_cos 0.01 --torso_joint 13 --eval_only  && echo "EXP 3 Launched, with learn adj,cos loss 0.01" &
# P0=$!
# wait $P0

# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --pred_seq_len 2 --use_cons --l_norm 0.01 --l_cos 0 --torso_joint 13 --eval_only  && echo "EXP 4 Launched, with learn adj,norm loss 0.01" &
# P0=$!
# wait $P0
# echo "0.25 time ends"

# #0.50 time --pred_seq_len 5
# echo "0.50 time start"
# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --learn_A --pred_seq_len 5 --torso_joint 13 --eval_only  && echo "EXP 0 Launched, standard model" &
# P0=$!
# wait $P0

# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --pred_seq_len 5 --torso_joint 13 --eval_only  && echo "EXP 1 Launched, with learn adj" &
# P0=$!
# wait $P0


# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --pred_seq_len 5 --use_cons --l_norm 0.01 --l_cos 0.01 --torso_joint 13 --eval_only  && echo "EXP 2 Launched, with learn adj,cons loss 0.01" &
# P0=$!
# wait $P0


# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --pred_seq_len 5 --use_cons --l_norm 0 --l_cos 0.01 --torso_joint 13 --eval_only  && echo "EXP 3 Launched, with learn adj,cos loss 0.01" &
# P0=$!
# wait $P0

# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --pred_seq_len 5 --use_cons --l_norm 0.01 --l_cos 0 --torso_joint 13 --eval_only  && echo "EXP 4 Launched, with learn adj,norm loss 0.01" &
# P0=$!
# wait $P0
# echo "0.50 time ends"

# #0.75 time --pred_seq_len 7
# echo "0.75 time start"
# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --learn_A --pred_seq_len 7 --torso_joint 13 --eval_only  && echo "EXP 0 Launched, standard model" &
# P0=$!
# wait $P0

# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --pred_seq_len 7 --torso_joint 13 --eval_only  && echo "EXP 1 Launched, with learn adj" &
# P0=$!
# wait $P0


# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --pred_seq_len 7 --use_cons --l_norm 0.01 --l_cos 0.01 --torso_joint 13 --eval_only  && echo "EXP 2 Launched, with learn adj,cons loss 0.01" &
# P0=$!
# wait $P0


# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --pred_seq_len 7 --use_cons --l_norm 0 --l_cos 0.01 --torso_joint 13 --eval_only  && echo "EXP 3 Launched, with learn adj,cos loss 0.01" &
# P0=$!
# wait $P0

# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --pred_seq_len 7 --use_cons --l_norm 0.01 --l_cos 0 --torso_joint 13 --eval_only  && echo "EXP 4 Launched, with learn adj,norm loss 0.01" &
# P0=$!
# wait $P0
# echo "0.75 time ends"


#These takes too much time, will do later 

# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --background_back  --torso_joint 13 --eval_only  && echo "EXP 5 Launched, with learn adj, background" &
# P0=$!
# wait $P0

# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --video_back  --torso_joint 13 --eval_only  && echo "EXP 6 Launched, with learn adj, video" &
# P0=$!
# wait $P0