# Create the directory for 2 split checkpoints
mkdir outputs/split0
mkdir outputs/split1
mkdir outputs/split2

# Run the main file for pretraining the splits
split_no=0
python main_pretrain.py --wandb_run_name pretrain${split_no} --wandb_project_name DLProject2Pretrain --partition_ckpt_directory outputs/split${split_no} --partition_num ${split_no} --gpus 2 --strategy ddp

split_no=1
python main_pretrain.py --wandb_run_name pretrain${split_no} --wandb_project_name DLProject2Pretrain --partition_ckpt_directory outputs/split${split_no} --partition_num ${split_no} --gpus 2 --strategy ddp

split_no=2
python main_pretrain.py --wandb_run_name pretrain${split_no} --wandb_project_name DLProject2Pretrain --partition_ckpt_directory outputs/split${split_no} --partition_num ${split_no} --gpus 2 --strategy ddp

# Create the fusion outcome checkpoint directory
mkdir outputs/split0_fusion_outcome
mkdir outputs/split1_fusion_outcome
mkdir outputs/split2_fusion_outcome
