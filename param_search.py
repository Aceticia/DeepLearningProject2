import subprocess
from itertools import product

hiddens_lst = [256, 512, 768, 1024]
temperature_lst = [-1, 0.5, 1, 2]

for _ in range(5):
    for hiddens, temperature in product(hiddens_lst, temperature_lst):
        subprocess.run(
            ["python", "main_fusion.py", "--hiddens", str(hiddens), "--temperature", str(temperature), "--gpus", "2", "--strategy", "ddp"]
        )

