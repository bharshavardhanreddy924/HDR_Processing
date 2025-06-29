# streamlit_hdr_mertens.py
import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import io

st.set_page_config(layout="wide")
st.title("📸 HDR Image Processor - Mertens Fusion")

st.markdown("Upload three images of the **same scene** with different exposures:")

# Upload three images
cols = st.columns(3)
labels = ["Low Exposure", "Mid Exposure", "High Exposure"]
uploaded_files = [None, None, None]

for i in range(3):
    with cols[i]:
        uploaded_files[i] = st.file_uploader(f"Upload {labels[i]} Image", type=["jpg", "jpeg", "png"], key=i)
        if uploaded_files[i] is not None:
            st.image(uploaded_files[i], caption=labels[i], use_column_width=True)

# HDR Processing
if st.button("⚙️ Process HDR (Mertens Fusion)"):
    try:
        # Read uploaded files into OpenCV format
        images = []
        for file in uploaded_files:
            if file is not None:
                image = Image.open(file).convert("RGB")
                image_np = np.array(image)
                image_cv = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)
                images.append(image_cv)

        if len(images) != 3:
            st.error("Please upload all three images.")
        else:
            # Mertens Fusion
            merge_mertens = cv.createMergeMertens()
            hdr_mertens = merge_mertens.process(images)
            hdr_mertens = np.clip(hdr_mertens * 255, 0, 255).astype('uint8')
            output_rgb = cv.cvtColor(hdr_mertens, cv.COLOR_BGR2RGB)
            output_pil = Image.fromarray(output_rgb)

            st.success("✅ HDR Image Processed Successfully!")
            st.image(output_pil, caption="Merged HDR Output", use_column_width=True)

            # Download Button
            buffer = io.BytesIO()
            output_pil.save(buffer, format="JPEG")
            st.download_button(
                label="📥 Download HDR Image",
                data=buffer.getvalue(),
                file_name="hdr_output.jpg",
                mime="image/jpeg"
            )

    except Exception as e:
        st.error(f"Error processing HDR image: {e}")
