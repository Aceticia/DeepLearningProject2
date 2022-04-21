for split_num in {30..50}
do
  # Create the directory for 2 split checkpoints
  mkdir outputs/split${split_num}

  # Run the main file for pretraining the splits
  for split_no in {0..29}
  do
    python main_pretrain.py --wandb_run_name split${split_num}_pretrain${split_no} --wandb_project_name DLProject2Pretrain --partition_ckpt_directory outputs/split${split_num} --partition_total_num ${split_num} --partition_num ${split_no}
  done

  # Create the fusion outcome checkpoint directory
  mkdir outputs/split${split_num}_fusion_outcome

  # Run the main for fusion of the splits
  for run_id in {1..5}
  do
    python main_fusion.py --wandb_run_name split${split_num}_run --wandb_project_name DLProject2Fusion --partition_ckpt_directory outputs/split${split_num}
  done
done
