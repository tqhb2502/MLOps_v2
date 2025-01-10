# Hệ thống dự đoán lương dựa trên kinh nghiệm
Hệ thống dự đoán lương dựa trên kinh nghiệm làm việc trong ngành công nghệ thống tin, giúp nhà tuyển dụng và ứng viên có sự tham khảo về mức lương hợp lý.
## Cài đặt
- Yêu cầu phiên bản `Python 3.12.8`
- Cài đặt các thư viện cần thiết:
`pip install -r requiments.txt`
## Huấn luyện
- Vào thư mục dự án:
`cd MLOps_v2`
- Tiến hành huấn luyện mô hình:
`python src/train.py`
## Chú ý
- Thêm HuggingFace write token vào `config.yaml` và `src/BE/main.py`
