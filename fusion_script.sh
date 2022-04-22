for split_num in {30..32}
do
  # Create the fusion outcome checkpoint directory
  mkdir outputs/split${split_num}_fusion_outcome

  for temperature in 1 2 5 10
  do
    # Run the main for fusion of the splits
    for run_id in {1..5}
    do
      python main_fusion.py --temperature ${temperature} --wandb_run_name split${split_num}_run --wandb_project_name DLProject2Fusion --partition_ckpt_directory outputs/split${split_num} --max_epochs 200 --fusion_outcome_ckpt_directory outputs/split${split_num}_fusion_outcome --gpus 2 --accelerator ddp
    done
  done
done
