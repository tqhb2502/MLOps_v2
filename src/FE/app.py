import streamlit as st
import requests

FEEDBACK_API_URL = "http://127.0.0.1:8000/colectdata"
PREDICT_API_URL = "http://127.0.0.1:8000/predict_salary" 

def predict_salary(experience_years):
    try:
        response = requests.post(
            PREDICT_API_URL, 
            json={"experience_years": experience_years}
        )
        response.raise_for_status()  # Kiểm tra nếu có lỗi
        result = response.json()
        return result.get("predicted_salary", "Lỗi: Không thể dự đoán lương!")
    except requests.exceptions.RequestException as e:
        return f"Lỗi khi gọi API: {e}"
# Hàm dịch mô phỏng

def collect_data(experience_years, predicted_salary):
    try:
        response = requests.post(
            FEEDBACK_API_URL, 
            json={"experience_years": experience_years, "predicted_salary": predicted_salary}
        )
        response.raise_for_status()  # Kiểm tra nếu có lỗi
        result = response.json()
        return result
    except requests.exceptions.RequestException as e:
        return f"Lỗi khi gọi API: {e}"

# Khởi tạo session state cho nút Like
if "liked" not in st.session_state:
    st.session_state.liked = False

st.title("Ứng dụng Dự đoán Lương Dựa trên Số Năm Kinh Nghiệm")

# Nhập vào số năm kinh nghiệm
experience_years = st.number_input("Nhập số năm kinh nghiệm", min_value=0, max_value=50, step=1)

if st.button("Dự đoán lương"):
    if experience_years <= 0:
        st.warning("Vui lòng nhập số năm kinh nghiệm hợp lệ!")
    else:
        predicted_salary = predict_salary(experience_years)
        st.session_state.predicted_salary = predicted_salary  # Lưu kết quả vào session state
        st.session_state.liked = False  # Reset trạng thái nút Like
        st.success(f"Dự đoán lương: {predicted_salary} VND")

# Hiển thị kết quả dự đoán lương
if "predicted_salary" in st.session_state:
    st.text_area("Kết quả dự đoán lương", st.session_state.predicted_salary, height=68, disabled=True)

# Nút Like và thu thập dữ liệu
if st.button("👍 Like", disabled=st.session_state.liked):
    collect_status = collect_data(experience_years, st.session_state.predicted_salary)
    st.session_state.liked = True
    st.write("Cảm ơn bạn đã thích dự đoán lương! 😊")
    st.write(f"Trạng thái thu thập dữ liệu: {collect_status}")