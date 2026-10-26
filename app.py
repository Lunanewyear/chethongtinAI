import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
import urllib.request
import os

st.title("Ứng dụng Che Mặt AI (Bản Cao Cấp - YuNet)")
st.write("Sử dụng AI thế hệ mới nhất để quét chính xác đám đông, công nhân ở xa mà không nhận nhầm đồ vật.")

blur_level = st.slider("Chọn mức độ làm mờ:", min_value=5, max_value=99, value=35, step=2)

uploaded_files = st.file_uploader("Chọn các bức ảnh (JPG, PNG)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Tự động tải mô hình YuNet siêu nhạy của OpenCV (Chỉ khoảng 1MB)
model_path = "face_detection_yunet_2023mar.onnx"
if not os.path.exists(model_path):
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    urllib.request.urlretrieve(url, model_path)

if uploaded_files:
    st.success(f"Đã tải lên {len(uploaded_files)} bức ảnh. Đang quét bằng AI YuNet...")
    
    for i, uploaded_file in enumerate(uploaded_files):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape
        
        # Khởi tạo YuNet quét ở độ phân giải gốc của bức ảnh
        detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(w, h),
            score_threshold=0.5,  # Mức độ tự tin (0.5 là vừa đủ để bắt mặt nhỏ mà không bắt nhầm vật)
            nms_threshold=0.3,
            top_k=5000
        )
        
        # Nhận diện
        _, faces = detector.detect(image)
        
        processed_image = image_rgb.copy()
        count_faces = 0
        
        # Vẽ vùng mờ nếu tìm thấy khuôn mặt
        if faces is not None:
            for face in faces:
                box = face[0:4].astype(np.int32)
                x, y, w_box, h_box = box
                
                # Tránh lỗi tràn viền ảnh
                x = max(0, x)
                y = max(0, y)
                w_box = min(w - x, w_box)
                h_box = min(h - y, h_box)
                
                if w_box <= 0 or h_box <= 0:
                    continue
                
                # Tạo hình bầu dục ôm sát
                center_x = x + w_box // 2
                center_y = y + h_box // 2
                radius_x = int(w_box * 0.45)
                radius_y = int(h_box * 0.55)
                
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 255, -1)
                
                blur_kernel = blur_level * 2 + 1
                blurred_full = cv2.GaussianBlur(processed_image, (blur_kernel, blur_kernel), 30)
                np.copyto(processed_image, blurred_full, where=(mask[..., None] == 255))
                count_faces += 1

        st.write(f"--- **Ảnh {i+1}:** {uploaded_file.name} (Đã che chuẩn xác {count_faces} người)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_rgb, caption="Ảnh gốc", use_container_width=True)
        with col2:
            st.image(processed_image, caption="Ảnh đã che mặt", use_container_width=True)
            
            result_pil = Image.fromarray(processed_image)
            buf = io.BytesIO()
            result_pil.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="📥 Tải ảnh này về máy",
                data=byte_im,
                file_name=f"che_mat_{uploaded_file.name}",
                mime="image/jpeg",
                key=f"download_{i}"
            )