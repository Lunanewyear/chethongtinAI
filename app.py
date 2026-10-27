import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
import urllib.request
import os
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Hệ Thống Xóa AI", layout="wide")

st.title("🛡️ Hệ Thống Xóa AI")
st.write("Hệ thống thông minh chuyên hỗ trợ xóa mặt người, tài liệu, con dấu, chữ ký...")

# --- KHỞI TẠO BIẾN TRẠNG THÁI CHO THANH TRƯỢT ---
if 'blur_level' not in st.session_state:
    st.session_state.blur_level = 35

def increase_blur():
    st.session_state.blur_level = min(99, st.session_state.blur_level + 2)

def decrease_blur():
    st.session_state.blur_level = max(5, st.session_state.blur_level - 2)

# --- TÙY CHỌN RA MÀN HÌNH CHÍNH ---
col_opt1, col_opt2 = st.columns([2, 2])
with col_opt1:
    security_mode = st.radio(
        "Chọn chế độ AI tự động:",
        ("👤 1. Chỉ xóa khuôn mặt", "📄 2. Xóa hồ sơ (Mặt oval, Dấu tròn, Tên công ty & Dữ liệu cá nhân)"),
        horizontal=True
    )

with col_opt2:
    st.markdown("**Độ làm mờ tổng thể:**")
    c1, c2, c3 = st.columns([1, 8, 1])
    with c1:
        st.button("➖", on_click=decrease_blur, help="Giảm độ mờ", use_container_width=True)
    with c2:
        st.slider("Độ làm mờ", min_value=5, max_value=99, key="blur_level", step=2, label_visibility="collapsed")
    with c3:
        st.button("➕", on_click=increase_blur, help="Tăng độ mờ", use_container_width=True)

blur_level = st.session_state.blur_level
st.markdown("---")

uploaded_files = st.file_uploader(
    "Chọn các bức ảnh cần xóa/bảo mật",
    type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff", "jfif"],
    accept_multiple_files=True
)

model_path = "face_detection_yunet_2023mar.onnx"
if not os.path.exists(model_path):
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    urllib.request.urlretrieve(url, model_path)

if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            continue
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape
        processed_image = image_rgb.copy()
        
        # ==========================================
        # PHẦN 1: AI TỰ ĐỘNG XỬ LÝ
        # ==========================================
        max_size = 1024
        scale = 1.0
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            detect_w, detect_h = int(w * scale), int(h * scale)
            detect_img = cv2.resize(image_bgr, (detect_w, detect_h))
        else:
            detect_img, detect_h, detect_w = image_bgr, h, w

        detector = cv2.FaceDetectorYN.create(
            model=model_path, config="", input_size=(detect_w, detect_h),
            score_threshold=0.5, nms_threshold=0.3, top_k=5000
        )
        _, faces = detector.detect(detect_img)
        
        if faces is not None and len(faces) > 0:
            for face in faces:
                box = (face[0:4] / scale).astype(np.int32)
                x, y, w_box, h_box = max(0, box[0]), max(0, box[1]), min(w - max(0, box[0]), box[2]), min(h - max(0, box[1]), box[3])
                if w_box > 0 and h_box > 0 and (w_box * h_box) < (w * h * 0.05):
                    cx, cy = x + w_box // 2, y + h_box // 2
                    rx, ry = int(w_box * 0.42), int(h_box * 0.50)
                    mask_face = np.zeros((h, w), dtype=np.uint8)
                    cv2.ellipse(mask_face, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
                    blur_kernel = blur_level * 2 + 1
                    blurred_full = cv2.GaussianBlur(processed_image, (blur_kernel, blur_kernel), 30)
                    processed_image[mask_face == 255] = blurred_full[mask_face == 255]

        if security_mode == "📄 2. Xóa hồ sơ (Mặt oval, Dấu tròn, Tên công ty & Dữ liệu cá nhân)":
            try:
                temp_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                hsv = cv2.cvtColor(temp_bgr, cv2.COLOR_BGR2HSV)

                # Mộc đỏ
                mask_r = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])) | cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
                margin = int(w * 0.02)
                mask_r[:, :margin] = 0; mask_r[:, w-margin:] = 0; mask_r[:margin, :] = 0; mask_r[h-margin:, :] = 0
                contours_r, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mask_stamp = np.zeros((h, w), dtype=np.uint8)
                for cnt in contours_r:
                    if cv2.contourArea(cnt) > 200: 
                        cv2.drawContours(mask_stamp, [cv2.convexHull(cnt)], -1, 255, -1)
                mask_stamp = cv2.dilate(mask_stamp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)

                # Chữ ký xanh
                mask_b = cv2.morphologyEx(cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255])), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
                contours_b, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mask_sign = np.zeros((h, w), dtype=np.uint8)
                for cnt in contours_b:
                    if cv2.contourArea(cnt) > 100:
                        x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
                        exp_y, exp_x = int(h * 0.05), int(w * 0.02)
                        cv2.rectangle(mask_sign, (max(0, x_b - exp_x), max(0, y_b - exp_x)), (min(w, x_b + w_b + exp_x), min(h, y_b + h_b + exp_y)), 255, -1)

                mask_combined = mask_stamp | mask_sign
                if np.any(mask_combined > 0):
                    bk = blur_level * 2 + 1
                    processed_image[mask_combined == 255] = cv2.GaussianBlur(processed_image, (bk, bk), 30)[mask_combined == 255]

                # Vùng text tĩnh (đã bỏ tiêu đề)
                precise_cert_boxes = [
                    (int(w * 0.21), int(h * 0.210), int(w * 0.40), int(h * 0.025)), 
                    (int(w * 0.24), int(h * 0.246), int(w * 0.60), int(h * 0.026)), 
                    (int(w * 0.13), int(h * 0.282), int(w * 0.45), int(h * 0.026)), 
                    (int(w * 0.21), int(h * 0.688), int(w * 0.40), int(h * 0.025)), 
                    (int(w * 0.24), int(h * 0.724), int(w * 0.60), int(h * 0.026)), 
                    (int(w * 0.13), int(h * 0.760), int(w * 0.45), int(h * 0.026)), 
                ]
                for (bx, by, bw, bh) in precise_cert_boxes:
                    bx, by, bw, bh = max(0, bx), max(0, by), min(w - bx, bw), min(h - by, bh)
                    if bw > 0 and bh > 0:
                        roi = processed_image[by:by+bh, bx:bx+bw]
                        if roi.size > 0:
                            processed_image[by:by+bh, bx:bx+bw] = cv2.GaussianBlur(roi, (blur_level * 2 + 1, blur_level * 2 + 1), 30)
            except Exception as e:
                pass

        # ==========================================
        # PHẦN 2: CÔNG CỤ CHỈNH SỬA THỦ CÔNG (ĐA GIÁC LỘN XỘN)
        # ==========================================
        final_image = processed_image.copy()
        
        st.markdown(f"### 🖼️ Xử lý ảnh: {uploaded_file.name}")
        
        with st.expander("🛠️ CÔNG CỤ KHOANH VÙNG ĐA GIÁC (Nhấn để mở nếu AI che sót/che nhầm)", expanded=False):
            st.info("💡 **Cách dùng:** Click chuột vào các điểm để nối thành hình khối (hình thang, lục giác...). **Click ĐÚP (2 lần)** hoặc click lại vào điểm đầu tiên để đóng/kết thúc hình!")
            edit_mode = st.radio(
                "Chọn hành động:",
                ("🟦 1. Vẽ vùng để THÊM MỜ", "🟥 2. Vẽ vùng để KHÔI PHỤC ẢNH GỐC"),
                horizontal=True, key=f"tool_{i}"
            )

            # Cài đặt màu sắc bọc hình để phân biệt dễ dàng
            if "THÊM" in edit_mode:
                stroke_color = "rgba(0, 0, 255, 1)" # Viền xanh đậm
                fill_color = "rgba(0, 0, 255, 0.4)" # Lõi xanh mờ
            else:
                stroke_color = "rgba(255, 0, 0, 1)" # Viền đỏ đậm
                fill_color = "rgba(255, 0, 0, 0.4)" # Lõi đỏ mờ

            canvas_w = 800
            canvas_h = int((h / w) * canvas_w)
            
            # Khởi tạo Canvas với chế độ Đa giác (Polygon)
            canvas_result = st_canvas(
                fill_color=fill_color, 
                stroke_width=2,
                stroke_color=stroke_color,
                background_image=Image.fromarray(processed_image),
                update_streamlit=True,
                height=canvas_h,
                width=canvas_w,
                drawing_mode="polygon", # ĐỔI SANG CHẾ ĐỘ ĐA GIÁC (TỰ DO)
                return_image_data=True, # BẮT BUỘC để giải quyết lỗi cập nhật ảnh
                key=f"canvas_{i}",
            )

            # --- LOGIC XỬ LÝ ẢNH MỚI BẰNG ĐIỂM ẢNH (PIXEL MASK) ---
            if canvas_result.image_data is not None:
                # Lấy bản đồ vùng đã vẽ (Resize về kích thước ảnh gốc)
                mask = cv2.resize(canvas_result.image_data, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # RGBA Format: 0=Red, 1=Green, 2=Blue, 3=Alpha
                # Lọc vùng Vẽ Xanh (Thêm Mờ): Kênh Blue cao, Red thấp
                blue_mask = (mask[:, :, 2] > 100) & (mask[:, :, 0] < 100) & (mask[:, :, 3] > 0)
                if np.any(blue_mask):
                    blur_k = blur_level * 2 + 1
                    blurred = cv2.GaussianBlur(image_rgb, (blur_k, blur_k), 30)
                    final_image[blue_mask] = blurred[blue_mask]

                # Lọc vùng Vẽ Đỏ (Khôi phục gốc): Kênh Red cao, Blue thấp
                red_mask = (mask[:, :, 0] > 100) & (mask[:, :, 2] < 100) & (mask[:, :, 3] > 0)
                if np.any(red_mask):
                    final_image[red_mask] = image_rgb[red_mask]

        # ==========================================
        # PHẦN 3: HIỂN THỊ KẾT QUẢ SO SÁNH & TẢI XUỐNG
        # ==========================================
        st.markdown("**So sánh kết quả:**")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.image(Image.fromarray(image_rgb), caption="🖼️ Ảnh Gốc", use_container_width=True)
            
        with col_right:
            final_pil = Image.fromarray(final_image)
            st.image(final_pil, caption="✨ Ảnh Hiện Tại (Kết quả)", use_container_width=True)
            
            ext = uploaded_file.name.split(".")[-1].lower()
            buf = io.BytesIO()
            if ext in ["png"]: final_pil.save(buf, format="PNG")
            elif ext in ["webp"]: final_pil.save(buf, format="WEBP")
            elif ext in ["bmp"]: final_pil.save(buf, format="BMP")
            elif ext in ["tif", "tiff"]: final_pil.save(buf, format="TIFF")
            else: final_pil.save(buf, format="JPEG", quality=95)
                
            st.download_button(
                label="📥 TẢI ẢNH HOÀN CHỈNH NÀY",
                data=buf.getvalue(),
                file_name=f"secure_{uploaded_file.name}",
                mime="application/octet-stream",
                key=f"download_{i}",
                use_container_width=True
            )
            
        st.markdown("---")
