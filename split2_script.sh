for split_num in {30..50}
do
  # Create the directory for 2 split checkpoints
  mkdir outputs/split${split_num}

  # Run the main file for pretraining the splits
  split_no=0
  while [ "$split_no" -lt "$split_num" ]; do
    python main_pretrain.py --wandb_run_name split${split_num}_pretrain${split_no} --wandb_project_name DLProject2Pretrain --partition_ckpt_directory outputs/split${split_num} --partition_total_num ${split_num} --partition_num ${split_no} --gpus 2 --accelerator ddp
    split_no=$(($split_no + 1))
  done

  # Create the fusion outcome checkpoint directory
  mkdir outputs/split${split_num}_fusion_outcome

  # Run the main for fusion of the splits
  for run_id in {1..5}
  do
    python main_fusion.py --wandb_run_name split${split_num}_run --wandb_project_name DLProject2Fusion --partition_ckpt_directory outputs/split${split_num} --max_epochs 200 --fusion_outcome_ckpt_directory outputs/split${split_num}_fusion_outcome --gpus 2 --accelerator ddp
  done
done
