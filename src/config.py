import os
import yaml

with open(os.path.join("config.yaml"), 'r') as file:
    config = yaml.safe_load(file)

DATA_CONFIG = config["data"]
SYSTEM_CONFIG = config["system"]
MODEL_CONFIG = config["model"]
MSG_CONFIG = config["message"]
LOG_CONFIG = config["log"]