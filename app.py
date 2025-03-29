import streamlit as st
import cv2
import numpy as np
import tempfile

# Function to enhance underwater images
def enhance_image(image):
    # Convert to LAB color space for color correction
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge LAB channels and convert back to BGR
    enhanced_lab = cv2.merge((l, a, b))
    color_corrected = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Apply White Balancing
  
    # Apply White Balancing using manual method
    balanced = simple_white_balance(color_corrected)


    # Apply Noise Reduction
    final_image = cv2.fastNlMeansDenoisingColored(balanced, None, 7, 7, 5, 15)

    return final_image
def simple_white_balance(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(result)
    l = cv2.equalizeHist(l)  # Apply histogram equalization
    result = cv2.merge((l, a, b))
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
 
# Streamlit UI
st.title("🌊 Underwater Image Enhancement App")
st.write("Upload an underwater image, and the AI will enhance its clarity!")

# File uploader
uploaded_file = st.file_uploader("Choose an underwater image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display original image
    st.image(image, caption="Original Image", use_container_width=True)

    # Enhance the image
    enhanced_image = enhance_image(image)

    # Display enhanced image
    st.image(enhanced_image, caption="Enhanced Image", use_container_width=True)

    # Download button for the enhanced image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        cv2.imwrite(temp_file.name, enhanced_image)
        st.download_button(
            label="📥 Download Enhanced Image",
            data=open(temp_file.name, "rb").read(),
            file_name="enhanced_underwater.jpg",
            mime="image/jpeg"
        )
