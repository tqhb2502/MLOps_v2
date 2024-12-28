import os
import logging
import matplotlib.pyplot as plt
from config import LOG_CONFIG
import utils

def evaluate_model(regressor, X_train, y_train, X_test, y_test):
    # Tạo thư mục logs nếu chưa tồn tại
    os.makedirs(LOG_CONFIG["log-dir"], exist_ok=True)
    sub_log_dir = utils.create_unique_directory(LOG_CONFIG["log-dir"])
    log_file_path = os.path.join(sub_log_dir, "log.txt")

    # Thiết lập logging
    with open(log_file_path, "w"):
        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
    
    # Dự đoán
    y_pred = regressor.predict(X_test)
    
    # Ghi log
    logging.info("Evaluation started.")
    logging.info(f"Predicted values: {y_pred}")
    logging.info(f"True values: {y_test}")
    
    # Lưu đồ thị Training Set
    plt.figure()
    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.title('Salary vs Experience (Training Set)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    train_plot_path = os.path.join(
        sub_log_dir,
        "training_set_plot.png"
    )
    plt.savefig(train_plot_path)
    plt.close()
    logging.info(f"Training set plot saved at {train_plot_path}.")
    
    # Lưu đồ thị Test Set
    plt.figure()
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.title('Salary vs Experience (Test Set)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    test_plot_path = os.path.join(
        sub_log_dir,
        "test_set_plot.png"
    )
    plt.savefig(test_plot_path)
    plt.close()
    logging.info(f"Test set plot saved at {test_plot_path}.")
    
    logging.info("Evaluation completed.")
