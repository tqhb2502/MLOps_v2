import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

from config import DATA_CONFIG, SYSTEM_CONFIG, MODEL_CONFIG, MSG_CONFIG
from evaluate import evaluate_model

# Lấy config
DATA_SOURCE = DATA_CONFIG['source']
DATA_DIR = DATA_CONFIG['data-dir']
TRAIN_RATIO = DATA_CONFIG['train-ratio']
TEST_RATIO = DATA_CONFIG['test-ratio']

RANDOM_STATE = SYSTEM_CONFIG['random-state']

MODEL_DIR = MODEL_CONFIG['model-dir']
MODEL_NAME = MODEL_CONFIG['model-name']
SAVE_PROTOCOL = MODEL_CONFIG['save-protocol']

MODEL_LOADED = MSG_CONFIG['model-loaded']
NEW_MODEL = MSG_CONFIG['new-model']

dataset = pd.read_csv(os.path.join(DATA_DIR, DATA_SOURCE))
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size = TEST_RATIO/TRAIN_RATIO, 
    random_state = RANDOM_STATE
)

# Đường dẫn tệp
file_path = os.path.join(MODEL_DIR, MODEL_NAME)

# Kiểm tra tệp tồn tại
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        regressor = pickle.load(f)
    print(MODEL_LOADED)
else:
    regressor = LinearRegression()
    print(NEW_MODEL)

regressor.fit(X_train, y_train)

# Tạo thư mục nếu chưa tồn tại
os.makedirs(MODEL_DIR, exist_ok=True)

# Save model
with open(file_path, "wb") as f:
    pickle.dump(regressor, f, protocol=SAVE_PROTOCOL)

# Test & Ghi log
evaluate_model(regressor, X_train, y_train, X_test, y_test)
