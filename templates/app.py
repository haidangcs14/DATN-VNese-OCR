import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import tensorflow.keras.backend as K
import os


# Load mô hình
model = tf.keras.models.load_model('./models/model.h5')
save_dir  = 'data_saved'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def main():
    st.markdown("""<h1 style="text-align: center; color: green">ĐỒ ÁN TỐT NGHIỆP</h1>""", unsafe_allow_html=True)
    st.markdown("""<h1 style="text-align: center">ĐỀ TÀI: XÂY DỰNG HỆ THỐNG NHẬN DẠNG KÍ TỰ QUANG HỌC SỬ DỤNG MẠNG NƠ-RON</h1>""", unsafe_allow_html=True)
    
    # st.title('ĐỀ TÀI: XÂY DỰNG HỆ THỐNG NHẬN DẠNG KÍ TỰ QUANG HỌC SỬ DỤNG MẠNG NƠ-RON')
    prediction = ''
    # Tạo layout với hai cột
    col1, col2 = st.columns([2, 1])

    with col1:
            uploaded_image = st.file_uploader("Upload hình ảnh", type=["jpg", "jpeg", "png"])
            if uploaded_image is None:
                st.warning("Vui lòng tải lên ảnh")
            else:
                image = Image.open(uploaded_image)
                st.image(image, caption="Hình ảnh đã tải lên", use_column_width=True)
                if st.button("Nhận dạng"):
                    # Xử lý ảnh và dự đoán
                    processed_image = preprocess_image(image)
                    prediction = predict_text(processed_image)

    # Cột bên phải - Kết quả dự đoán
    with col2:
        if uploaded_image is None:
            st.markdown(
                """
                <div style="text-align: center; border: 2px solid #FF0000; padding: 10px; background-color: #FFCCCC; margin-top:50px; width: 200px;">
                    <h3 style="color: #FF0000;">Kết quả nhận dạng</h3>
                    <p>Vui lòng tải lên ảnh để nhận dạng</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="text-align: center; border: 2px solid #008000; padding: 10px; background-color: #E6F4E6; margin-top:50px; width: 200px; heigh: 100px">
                    <h3 style="color: #008000;">Kết quả nhận dạng</h3>
                    <p>{}</p>
                </div>
                """.format(prediction),
                unsafe_allow_html=True
            )


def preprocess_image(image):
    # Chuyển đổi hình ảnh thành dạng NumPy array
    img_array = np.array(image)
    if img_array.ndim == 3:
        # Chuyển đổi hình ảnh thành grayscale
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img_array
    # Chỉnh kích thước hình ảnh
    resized_img = cv2.resize(gray_img, (2167, 118))

    img = np.pad(resized_img, ((0, 2167 - resized_img.shape[1]), (0, 0)), 'median')
    # Blur the image
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Threshold the image using adaptive thresholding
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    # Chuẩn hóa giá trị pixel
    normalized_img = img / 255.0
    # Thêm chiều batch (None) và chiều channel (1) vào hình ảnh
    processed_image = np.expand_dims(np.expand_dims(normalized_img, axis=2), axis=0)

    return processed_image

def predict_text(image):
    # Đưa hình ảnh vào mô hình để dự đoán văn bản
    predictions = model.predict(image)
    # Giải mã kết quả dự đoán thành văn bản
    decoded_text = decode_predictions(predictions)
    # Trả về kết quả dự đoán
    return decoded_text

def decode_predictions(prediction):
    # Sử dụng CTC decoder
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1], greedy=True)[0][0])

    # Danh sách ký tự
    char_list = "#'()+,-./0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvwxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ "

    # Hiển thị kết quả
    print("Kết quả dự đoán: ", end='')
    pred = ""
    for p in out[0]:
        if int(p) != -1:
            pred += char_list[int(p) - 1]
    print(pred)
    return pred

if __name__ == "__main__":
    main()
