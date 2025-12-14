import subprocess
import yaml

config_path = "config/test_config.yaml"

# 5 个随机种子
seeds = [0, 1, 42, 999, 2025]

# 8 个算法
algorithms = [
    "FedAvg",
    "FedAvgECGR",
    "FedProx",
    "FedProxECGR",
    "FedNova",
    "FedNovaECGR",
    "Scaffold",
    "ScaffoldECGR"
]

for s in seeds:
    for algo in algorithms:
        # 1. 读取 YAML
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # 2. 修改 seed
        config["system"]["i_seed"] = s

        # 3. 修改联邦算法
        config["client"]["fed_algo"] = algo

        # 4. 写回 YAML
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        print(f"\n===== Running with i_seed = {s}, fed_algo = {algo} =====")

        # 5. 运行 fl_main.py
        subprocess.run(["python", "fl_main.py", "--config", config_path])
