from huggingface_hub import Repository
import os
import shutil
from config import SYSTEM_CONFIG, MODEL_CONFIG, MSG_CONFIG

def push_model_to_huggingface(local_model_path, repo_name, token):
    """
    Pushes a local model to Hugging Face Hub.

    Args:
        local_model_path (str): Path to the local model directory.
        repo_name (str): Name of the Hugging Face repository.
        token (str): Hugging Face token.
    """
    # Tạo thư mục tạm nếu chưa có
    repo_local_dir = f"./{repo_name}"
    
    # Khởi tạo repo hoặc clone repo nếu đã tồn tại
    repo = Repository(
        local_dir=repo_local_dir, 
        clone_from=f"{repo_name}", 
        use_auth_token=token
    )
    
    # Xóa model cũ trong repo (nếu có)
    model_file_path = os.path.join(repo_local_dir, MODEL_CONFIG["model-name"])
    if os.path.exists(model_file_path):
        os.remove(model_file_path)
    
    # Sao chép mô hình vào thư mục repo
    new_model_path = os.path.join(local_model_path, MODEL_CONFIG["model-name"])
    shutil.copy(new_model_path, model_file_path)
    
    # Push lên Hugging Face
    repo.push_to_hub(commit_message=MSG_CONFIG["huggingface-commit"])
    print(MSG_CONFIG["huggingface-pushed"] + repo_name)

# Sử dụng hàm
local_model_path = MODEL_CONFIG["model-dir"]  # Đường dẫn thư mục chứa mô hình
repo_name = SYSTEM_CONFIG["huggingface-repo"]  # Tên repo (thay 'username' và 'model_name' bằng của bạn)
token = SYSTEM_CONFIG["huggingface-token"]  # Token của bạn

push_model_to_huggingface(local_model_path, repo_name, token)
