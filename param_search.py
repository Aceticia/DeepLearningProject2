import subprocess
from itertools import product

hiddens_lst = [256, 512, 768, 1024]
temperature_lst = [-1, 0.5, 1, 2]
loss_func_lst = ["SmoothL1Loss", "CosineSimilarity", "KLDivLoss"]

for _ in range(5):
    for hiddens, temperature, loss_func in product(hiddens_lst, temperature_lst, loss_func_lst):
        subprocess.run(
            ["python", "main_fusion.py", "--hiddens", str(hiddens), "--temperature", str(temperature), "--loss_type", loss_func, "--gpus", "2", "--strategy", "ddp"]
        )

