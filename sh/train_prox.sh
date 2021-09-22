# !/bin/bash
echo " Running PROX Training EXP"

#Full time --pred_seq_len 10
echo "Full time start"
CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.03 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --learn_A   && echo "EXP 0 Launched, standard model" &
P0=$!
wait $P0

CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.03 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450   && echo "EXP 1 Launched, with learn adj" &
P0=$!
wait $P0


CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.03 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.005 --l_cos 0.005   && echo "EXP 2 Launched, with learn adj,cons loss 0.01" &
P0=$!
wait $P0


CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.03 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0 --l_cos 0.005    && echo "EXP 3 Launched, with learn adj,cos loss 0.01" &
P0=$!
wait $P0

CUDA_VISIBLE_DEVICES=3 python3 train.py --lr 0.03 --n_stgcnn 5 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.005 --l_cos 0   && echo "EXP 4 Launched, with learn adj,norm loss 0.01" &
P0=$!
wait $P0
echo "Full time ends"


# echo "Full time start2"
# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.03 --n_stgcnn 3 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --learn_A   --tag 2 --torso_joint 13 --eval_only && echo "EXP 0 Launched, standard model" &
# P0=$!
# wait $P0

# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.03 --n_stgcnn 3 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450    --tag 2 --torso_joint 13 --eval_only && echo "EXP 1 Launched, with learn adj" &
# P0=$!
# wait $P0


# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.03 --n_stgcnn 3 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.005 --l_cos 0.005   --tag 2 --torso_joint 13 --eval_only && echo "EXP 2 Launched, with learn adj,cons loss 0.01" &
# P0=$!
# wait $P0


# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.03 --n_stgcnn 3 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0 --l_cos 0.005   --tag 2 --torso_joint 13 --eval_only && echo "EXP 3 Launched, with learn adj,cos loss 0.01" &
# P0=$!
# wait $P0

# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.03 --n_stgcnn 3 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.005 --l_cos 0   --tag 2 --torso_joint 13 --eval_only && echo "EXP 4 Launched, with learn adj,norm loss 0.01" &
# P0=$!
# wait $P0
# echo "Full time ends2"

# echo "Full time start3"
# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.03 --n_stgcnn 3 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --learn_A    --tag 3 --torso_joint 13 --eval_only && echo "EXP 0 Launched, standard model" &
# P0=$!
# wait $P0

# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.03 --n_stgcnn 3 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450    --tag 3 --torso_joint 13 --eval_only && echo "EXP 1 Launched, with learn adj" &
# P0=$!
# wait $P0


# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.03 --n_stgcnn 3 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.005 --l_cos 0.005   --tag 3 --torso_joint 13 --eval_only && echo "EXP 2 Launched, with learn adj,cons loss 0.01" &
# P0=$!
# wait $P0


# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.03 --n_stgcnn 3 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0 --l_cos 0.005   --tag 3 --torso_joint 13 --eval_only && echo "EXP 3 Launched, with learn adj,cos loss 0.01" &
# P0=$!
# wait $P0

# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.03 --n_stgcnn 3 --n_txpcnn 11  --dataset PROX --use_lrschd --num_epochs 450 --use_cons --l_norm 0.005 --l_cos 0   --tag 3 --torso_joint 13 --eval_only && echo "EXP 4 Launched, with learn adj,norm loss 0.01" &
# P0=$!
# wait $P0
# echo "Full time ends3"



#These takes too much time, will do later 

# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --background_back  && echo "EXP 5 Launched, with learn adj, background" &
# P0=$!
# wait $P0

# CUDA_VISIBLE_DEVICES=2 python3 train.py --lr 0.01 --n_stgcnn 5 --n_txpcnn 11  --dataset GTA_IM --use_lrschd --num_epochs 450 --video_back  && echo "EXP 6 Launched, with learn adj, video" &
# P0=$!
# wait $P0