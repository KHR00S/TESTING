import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os
import cv2

# Load the ONNX model
onnx_model_path = 'best.onnx'
onnx_session = ort.InferenceSession(onnx_model_path)

# Define class labels
class_labels = ['Perubahan Warna Gigi', 'Radang Gusi', 'Gigi Berlubang', 'Gigi Sehat', 'Bukan Gigi']

# Function to preprocess image
def preprocess_image(image):
    # Resize image to (416, 416)
    image = image.resize((416, 416))
    # Convert image to numpy array
    img_array = np.array(image)
    # Transpose array to (channels, height, width)
    img_array = np.transpose(img_array, (2, 0, 1))
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize pixel values
    img_array = img_array.astype(np.float32) / 255.0
    return img_array

# Function to classify image
def classify_image(image):
    # Preprocess image
    img_array = preprocess_image(image)

    # Perform inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    classes = onnx_session.run([output_name], {input_name: img_array})[0]
    predicted_class_index = np.argmax(classes)

    # Get the predicted label and confidence
    if predicted_class_index < len(class_labels):
        predicted_label = class_labels[predicted_class_index]
        confidence = classes[0][predicted_class_index]
    else:
        predicted_label = "Kelas tidak dikenal"
        confidence = 0.0

    return predicted_label, confidence

# Main Streamlit app
def main():
    st.title("Deteksi Kesehatan Gigi")

    # Add instructions for the user
    st.write("Silakan aktifkan kamera untuk memfoto gigi bagian depan Anda.")

    # Capture image from webcam
    img = st.image([], channels='RGB')

    # Button to capture image
    capture_button = st.button("Ambil Foto")

    if capture_button:
        # Capture image from webcam
        captured_img = st.empty()
        camera = st.empty()

        # Turn on webcam
        cap = cv2.VideoCapture(0)

        # Read and display frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Convert frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert frame to PIL Image
            pil_image = Image.fromarray(frame_rgb)

            # Display the camera view
            camera.image(frame_rgb, channels='RGB', use_column_width=True)

            # Button to confirm capturing the image
            confirm_capture_button = st.button("Konfirmasi")

            if confirm_capture_button:
                # Classify the captured image
                predicted_label, confidence = classify_image(pil_image)

                # Display the prediction
                captured_img.image(frame_rgb, channels='RGB', use_column_width=True)
                st.write(f"Prediksi: {predicted_label} (Kepercayaan: {confidence:.2f})")

                # Release the webcam
                cap.release()
        else:
            st.error("Tidak dapat membaca gambar dari kamera.")

if __name__ == "__main__":
    main()
