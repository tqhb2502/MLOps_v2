import os
import random
import string
from datetime import datetime

def create_unique_directory(base_path="."):
    # Lấy ngày hiện tại
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Tạo 5 ký tự ngẫu nhiên
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    
    # Tên thư mục
    folder_name = f"{current_date}_{random_suffix}"
    
    # Đường dẫn đầy đủ
    full_path = os.path.join(base_path, folder_name)
    
    # Tạo thư mục
    os.makedirs(full_path, exist_ok=True)
    
    return full_path
