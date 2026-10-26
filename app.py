import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
import urllib.request
import os

st.set_page_config(page_title="Hệ Thống Bảo Mật Ảnh & Hồ Sơ AI", layout="wide")

st.title("🛡️ Hệ Thống Bảo Mật Ảnh & Hồ Sơ AI")
st.write("Đã tối ưu hóa vùng mờ con dấu: **Dùng khuôn hình tròn/elip ôm trọn trọn vẹn viền con dấu đỏ**, đảm bảo tính thẩm mỹ và bảo mật tuyệt đối.")

# --- TÙY CHỌN RA MÀN HÌNH CHÍNH ---
col_opt1, col_opt2 = st.columns([2, 2])
with col_opt1:
    security_mode = st.radio(
        "Chọn chế độ bảo mật:",
        ("👤 1. Chỉ che khuôn mặt", "📄 2. Che hồ sơ (Mặt, Dấu tròn trọn vẹn, Tên công ty & Dữ liệu cá nhân)"),
        horizontal=True
    )

with col_opt2:
    blur_level = st.slider("Độ làm mờ:", min_value=5, max_value=99, value=35, step=2)

st.markdown("---")

uploaded_files = st.file_uploader(
    "Chọn các bức ảnh cần bảo mật",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp",
        "tif",
        "tiff",
        "jfif"
    ],
    accept_multiple_files=True
)

# Tải mô hình YuNet của OpenCV
model_path = "face_detection_yunet_2023mar.onnx"
if not os.path.exists(model_path):
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    urllib.request.urlretrieve(url, model_path)

if uploaded_files:
    st.success(f"Đã tải lên {len(uploaded_files)} bức ảnh. Đang tiến hành xử lý...")
    
    for i, uploaded_file in enumerate(uploaded_files):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            continue
            
        # Chuyển sang RGB chuẩn xác
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape
        processed_image = image_rgb.copy()
        
        count_faces = 0
        
        # --- 1. LUÔN QUÉT VÀ LÀM MỜ KHUÔN MẶT / ẢNH THẺ ---
        max_size = 1024
        scale = 1.0
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            detect_w = int(w * scale)
            detect_h = int(h * scale)
            detect_img = cv2.resize(image_bgr, (detect_w, detect_h))
        else:
            detect_img = image_bgr
            detect_h, detect_w = h, w

        detector = cv2.FaceDetectorYN.create(
            model=model_path, config="", input_size=(detect_w, detect_h),
            score_threshold=0.5, nms_threshold=0.3, top_k=5000
        )
        _, faces = detector.detect(detect_img)
        
        if faces is not None and len(faces) > 0:
            for face in faces:
                box = face[0:4] / scale
                box = box.astype(np.int32)
                x, y, w_box, h_box = box
                x, y = max(0, x), max(0, y)
                w_box = min(w - x, w_box)
                h_box = min(h - y, h_box)
                
                if w_box > 0 and h_box > 0:
                    face_area = w_box * h_box
                    image_area = w * h
                    
                    if face_area < image_area * 0.05:
                        if w_box < w * 0.4 and h_box < h * 0.4:
                            roi = processed_image[y:y+h_box, x:x+w_box]
                            if roi.size > 0:
                                processed_image[y:y+h_box, x:x+w_box] = cv2.GaussianBlur(roi, (blur_level * 2 + 1, blur_level * 2 + 1), 30)
                        else:
                            center_x, center_y = x + w_box // 2, y + h_box // 2
                            radius_x, radius_y = int(w_box * 0.45), int(h_box * 0.55)
                            mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 255, -1)
                            
                            blur_kernel = blur_level * 2 + 1
                            blurred_full = cv2.GaussianBlur(processed_image, (blur_kernel, blur_kernel), 30)
                            processed_image[mask == 255] = blurred_full[mask == 255]
                        count_faces += 1

        # --- 2. NẾU CHỌN CHẾ ĐỘ "CHE HỒ SƠ" -> TỰ ĐỘNG BẮT DẤU TRÒN (BỌC TRÒN TUYỆT ĐỐI) & CĂN CHỈNH TỌA ĐỘ ---
        if security_mode == "📄 2. Che hồ sơ (Mặt, Dấu tròn trọn vẹn, Tên công ty & Dữ liệu cá nhân)":
            try:
                if h > w:
                    # A. TỰ ĐỘNG NHẬN DIỆN VÀ BỌC TRÒN CON DẤU ĐỎ
                    temp_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                    hsv = cv2.cvtColor(temp_bgr, cv2.COLOR_BGR2HSV)

                    mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([12, 255, 255]))
                    mask2 = cv2.inRange(hsv, np.array([168, 50, 50]), np.array([180, 255, 255]))
                    mask_r = mask1 | mask2
                    
                    margin = int(w * 0.05)
                    mask_r[:, :margin] = 0
                    mask_r[:, w-margin:] = 0
                    mask_r[:margin, :] = 0
                    mask_r[h-margin:, :] = 0

                    kernel_red = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, kernel_red)
                    contours_r, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for cnt in contours_r:
                        area = cv2.contourArea(cnt)
                        if 300 < area < 15000:
                            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                            if radius > 12:
                                # Mở rộng bán kính thêm 15% để bọc trọn vẹn toàn bộ viền con dấu
                                stamp_radius = int(radius * 1.25)
                                center_x, center_y = int(cx), int(cy)
                                
                                mask_stamp = np.zeros((h, w), dtype=np.uint8)
                                cv2.circle(mask_stamp, (center_x, center_y), stamp_radius, 255, -1)
                                
                                blur_kernel = blur_level * 2 + 1
                                blurred_full = cv2.GaussianBlur(processed_image, (blur_kernel, blur_kernel), 30)
                                processed_image[mask_stamp == 255] = blurred_full[mask_stamp == 255]

                    # B. CÁC VÙNG THÔNG TIN TÊN CÔNG TY & DỮ LIỆU CÁ NHÂN
                    precise_cert_boxes = [
                        (int(w * 0.12), int(h * 0.035), int(w * 0.28), int(h * 0.032)),
                        (int(w * 0.12), int(h * 0.515), int(w * 0.28), int(h * 0.032)),
                        
                        (int(w * 0.28), int(h * 0.172), int(w * 0.55), int(h * 0.026)),
                        (int(w * 0.21), int(h * 0.210), int(w * 0.25), int(h * 0.025)),
                        (int(w * 0.24), int(h * 0.246), int(w * 0.60), int(h * 0.026)),
                        (int(w * 0.13), int(h * 0.282), int(w * 0.45), int(h * 0.026)),
                        
                        (int(w * 0.28), int(h * 0.650), int(w * 0.55), int(h * 0.026)),
                        (int(w * 0.21), int(h * 0.688), int(w * 0.25), int(h * 0.025)),
                        (int(w * 0.24), int(h * 0.724), int(w * 0.60), int(h * 0.026)),
                        (int(w * 0.13), int(h * 0.760), int(w * 0.45), int(h * 0.026)),
                    ]
                    
                    for (bx, by, bw, bh) in precise_cert_boxes:
                        bx, by = max(0, bx), max(0, by)
                        bw, bh = min(w - bx, bw), min(h - by, bh)
                        if bw > 0 and bh > 0:
                            roi = processed_image[by:by+bh, bx:bx+bw]
                            if roi.size > 0:
                                blurred_roi = cv2.GaussianBlur(roi, (blur_level * 2 + 1, blur_level * 2 + 1), 30)
                                processed_image[by:by+bh, bx:bx+bw] = blurred_roi
            except Exception as e:
                st.error(f"Lỗi xử lý hồ sơ: {str(e)}")

        st.write(f"--- **Ảnh {i+1}:** {uploaded_file.name} (Đã xử lý bảo mật thành công)")
        
        # --- HIỂN THỊ HÌNH ẢNH RA GIAO DIỆN TRỰC QUAN ---
        result_pil = Image.fromarray(processed_image)
        original_pil = Image.fromarray(image_rgb)

        col1, col2 = st.columns(2)
        with col1:
            st.image(original_pil, caption="Ảnh gốc", use_container_width=True)
        with col2:
            st.image(result_pil, caption=f"Kết quả ({security_mode})", use_container_width=True)
            
            ext = uploaded_file.name.split(".")[-1].lower()
            
            buf = io.BytesIO()
            if ext in ["png"]:
                result_pil.save(buf, format="PNG")
            elif ext in ["webp"]:
                result_pil.save(buf, format="WEBP")
            elif ext in ["bmp"]:
                result_pil.save(buf, format="BMP")
            elif ext in ["tif", "tiff"]:
                result_pil.save(buf, format="TIFF")
            else:
                result_pil.save(buf, format="JPEG", quality=95)
                
            byte_im = buf.getvalue()
            
            st.download_button(
                label="📥 Tải ảnh đã xử lý",
                data=byte_im,
                file_name=f"secure_{uploaded_file.name}",
                mime="application/octet-stream",
                key=f"download_{i}"
            )
