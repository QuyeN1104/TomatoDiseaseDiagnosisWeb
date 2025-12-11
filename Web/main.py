import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import os

# ==========================================
# 1. CẤU HÌNH & DATA
# ==========================================
MODEL_PATH = "cnn.h5"
CSV_PATH = "disease_database.csv"
SAMPLE_DIR = "test_images"  #

CLASS_NAMES = [
    'Bacterial spot', 'Early blight', 'Healthy', 'Late blight',
    'Leaf Mold', 'Septoria leaf spot', 'Spider mites',
    'Target Spot', 'Tomato mosaic virus', 'Yellow Leaf Curl Virus'
]
# ==========================================
# 2. HÀM LOAD (CACHE)
# ==========================================
@st.cache_resource
def load_prediction_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        return None


@st.cache_data
def load_database():
    try:
        return pd.read_csv(CSV_PATH, encoding="latin1")
    except:
        return None


def get_disease_info(df, disease_name):
    if df is None: return None
    search_term = disease_name.lower()
    if 'disease' not in df.columns: return None
    mask = df['disease'].str.lower().str.replace('_', ' ').str.contains(search_term, na=False)
    row = df[mask]
    if not row.empty:
        data = row.iloc[0]
        links_str = data['Link'] if 'Link' in df.columns and pd.notna(data['Link']) else ""
        controls_str = data['control'] if 'control' in df.columns and pd.notna(data['control']) else ""
        return {
            'description': data['description'] if 'description' in df.columns else "N/A",
            'treatments': [t.strip() for t in controls_str.split(';') if t.strip()],
            'links': [l.strip() for l in links_str.split(';') if l.strip()]
        }
    return None


# ==========================================
# 3. GIAO DIỆN CHÍNH
# ==========================================
st.set_page_config(page_title="Tomato Disease Diagnosis", layout="wide")

# --- KHỞI TẠO SESSION STATE ---
# Biến này giúp lưu giữ ảnh đang được chọn giữa các lần reload trang
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'image_source' not in st.session_state:
    st.session_state.image_source = ""  # Để biết ảnh đến từ 'upload' hay 'sample'

st.title("🍅 Ứng dụng Chẩn đoán Bệnh Cà Chua")
st.markdown("---")

model = load_prediction_model()
df_db = load_database()

# ==========================================
# SIDEBAR: CHỌN ẢNH MẪU (TEST IMAGES)
# ==========================================
with st.sidebar:
    st.header("📂 Thư viện ảnh mẫu")
    st.write("Click vào nút bên dưới để test nhanh:")

    # Kiểm tra xem thư mục ảnh mẫu có tồn tại không
    if os.path.exists(SAMPLE_DIR):
        # Lấy danh sách file ảnh
        sample_files = [f for f in os.listdir(SAMPLE_DIR) if f.endswith(('.JPG', '.png', '.jpeg'))]

        # Tạo lưới hiển thị ảnh nhỏ (2 cột)
        cols = st.columns(2)
        for i, file_name in enumerate(sample_files):
            file_path = os.path.join(SAMPLE_DIR, file_name)

            # Hiển thị ảnh nhỏ và nút chọn trong cột tương ứng
            with cols[i % 2]:
                try:
                    img_thumb = Image.open(file_path)
                    st.image(img_thumb)
                    if st.button(f"Chọn ảnh {i + 1}", key=f"btn_{i}"):
                        st.session_state.current_image = img_thumb
                        st.session_state.image_source = f"Mẫu: {file_name}"
                except:
                    pass
    else:
        st.warning(f"Chưa tạo thư mục '{SAMPLE_DIR}'")

# ==========================================
# KHUNG CHÍNH (MAIN PAGE)
# ==========================================
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📷 Input Hình Ảnh")

    # UPLOAD FILE
    uploaded_file = st.file_uploader("Hoặc tải lên ảnh của bạn:", type=["jpg", "png", "jpeg"])

    # Logic: Nếu người dùng upload file mới -> Ưu tiên hiển thị file upload
    if uploaded_file is not None:
        # Chỉ cập nhật nếu file upload khác với file đã lưu (tránh reload loop)
        # Ở đây ta đơn giản hóa: cứ có file upload là ưu tiên
        image_uploaded = Image.open(uploaded_file).convert("RGB")
        st.session_state.current_image = image_uploaded
        st.session_state.image_source = "Ảnh tải lên từ máy"

    # HIỂN THỊ ẢNH ĐANG ĐƯỢC CHỌN (Từ Session State)
    if st.session_state.current_image is not None:
        st.image(st.session_state.current_image, caption='Ảnh input')
        predict_btn = st.button("🔍 Chẩn đoán ngay", type="primary")
    else:
        st.info("👈 Hãy chọn ảnh mẫu bên trái hoặc tải ảnh lên.")
        predict_btn = False

with col2:
    st.subheader("📊 Kết quả Phân tích")

    if predict_btn and st.session_state.current_image is not None:
        if model is None:
            st.error("Lỗi: Model chưa được load.")
        else:
            with st.spinner("Đang chạy model AI..."):
                # Xử lý ảnh từ Session State
                img = st.session_state.current_image
                img_resized = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
                img_array = np.asarray(img_resized).astype(np.float32) / 255.0
                img_batch = np.expand_dims(img_array, axis=0)

                # Dự đoán
                prediction = model.predict(img_batch)
                predicted_idx = np.argmax(prediction)
                predicted_label = CLASS_NAMES[predicted_idx]
                confidence = np.max(prediction) * 100

                # Hiển thị
                st.success(f"Kết quả: **{predicted_label}**")
                st.metric("Độ tin cậy", f"{confidence:.2f}%")

                # Thông tin chi tiết
                st.markdown("---")
                info = get_disease_info(df_db, predicted_label)

                if info:
                    st.markdown(f"**📖 Mô tả:** {info['description']}")
                    st.markdown("**💊 Biện pháp xử lý:**")
                    pairs = list(zip(info['treatments'], info['links']))
                    if not pairs and info['treatments']:
                        for t in info['treatments']: st.markdown(f"- {t}")
                    for t, l in pairs:
                        st.markdown(f"- [{t}]({l})")
                else:
                    st.warning("Chưa có thông tin cách xử lý.")