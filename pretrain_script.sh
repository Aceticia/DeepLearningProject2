# Create the directory for 2 split checkpoints
mkdir ckpts

# Run the main file for pretraining the splits
split_no=0
python main_pretrain.py --wandb_run_name pretrain${split_no} --wandb_project_name DLProject2Pretrain --partition_ckpt_directory ckpts --partition_num ${split_no} --gpus 2 --strategy ddp

split_no=1
python main_pretrain.py --wandb_run_name pretrain${split_no} --wandb_project_name DLProject2Pretrain --partition_ckpt_directory ckpts --partition_num ${split_no} --gpus 2 --strategy ddp

split_no=2
python main_pretrain.py --wandb_run_name pretrain${split_no} --wandb_project_name DLProject2Pretrain --partition_ckpt_directory ckpts --partition_num ${split_no} --gpus 2 --strategy ddp

