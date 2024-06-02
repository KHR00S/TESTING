import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import cv2

# Load the ONNX model
onnx_model_path = 'best.onnx'
onnx_session = ort.InferenceSession(onnx_model_path)

# Define class labels
class_labels = ['Bukan Gigi', 'Gigi Berlubang', 'Gigi Sehat', 'Perubahan Warna Gigi', 'Radang Gusi']

# Function to preprocess image
def preprocess_image(image):
    # Resize image to (416, 416)
    image = image.resize((416, 416))
    # Convert image to numpy array
    img_array = np.array(image)
    # Transpose array to (channels, height, width)
    img_array = np.transpose(img_array, (2, 0, 1))
    # Normalize pixel values
    img_array = img_array.astype(np.float32) / 255.0
    return img_array

# Function to classify image
def classify_image(image):
    # Preprocess image
    img_array = preprocess_image(image)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Perform inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    classes = onnx_session.run([output_name], {input_name: img_array})[0]
    predicted_class_index = np.argmax(classes)
    # Get the predicted label and confidence
    predicted_label = class_labels[predicted_class_index]
    confidence = classes[0][predicted_class_index]
    return predicted_label, confidence

# Main Streamlit app
def main():
    st.title("Deteksi Kesehatan Gigi")
    # Sidebar with author information
    st.sidebar.header("Informasi Penulis")
    st.sidebar.text("Nama: FAKHRUS SYAKIR")
    st.sidebar.text("BANGKIT ID: M322D4KY1790")
    st.sidebar.text("GitHub: KHR00S")

    # Add instructions for the user
    st.write("Silakan pastikan gigi dan gusi Anda terlihat jelas di kamera. Hidung dan kumis harus tidak terlihat (bibir diperbolehkan).")

    # Button to capture image from camera
    if st.button("Ambil Gambar"):
        # OpenCV Capture
        cap = cv2.VideoCapture(0)

        # Read image from camera
        ret, frame = cap.read()
        
        if ret:
            # Convert frame to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Display captured image
            st.image(pil_image, caption="Gambar yang Diambil", use_column_width=True)

            try:
                # Classify the captured image
                predicted_label, confidence = classify_image(pil_image)
                # Display the prediction
                st.write(f"Prediksi: {predicted_label} (Kepercayaan: {confidence:.2f})")
            except Exception as e:
                st.write(f"An error occurred: {e}")
        else:
            st.write("Gagal mengambil gambar dari kamera")

        # Release the camera
        cap.release()

if __name__ == "__main__":
    main()
