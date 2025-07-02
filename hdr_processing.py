import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import io

# Page Configuration
st.set_page_config(page_title="HDR Image Processor", layout="wide")
st.title("📸 HDR Image Processor - Mertens Fusion")
st.markdown("Upload three images of the **same scene** with different exposures:")

# Resize function (reduce by 60%)
def resize_image_pil(pil_img, scale=0.4):
    width, height = pil_img.size
    return pil_img.resize((int(width * scale), int(height * scale)))

# Upload images in columns
labels = ["Low Exposure", "Mid Exposure", "High Exposure"]
uploaded_files = [None, None, None]
images_preview = [None, None, None]
cols = st.columns(3)

for i in range(3):
    with cols[i]:
        uploaded_files[i] = st.file_uploader(f"Upload {labels[i]} Image", type=["jpg", "jpeg", "png"], key=i)
        if uploaded_files[i]:
            image = Image.open(uploaded_files[i]).convert("RGB")
            resized = resize_image_pil(image)
            images_preview[i] = resized
            st.image(resized, caption=labels[i], use_container_width=True)  # ✅ Updated parameter

# HDR Processing
if st.button("⚙️ Process HDR (Mertens Fusion)"):
    if any(file is None for file in uploaded_files):
        st.error("❌ Please upload all three exposure images.")
    else:
        try:
            # Convert images to OpenCV format
            images_cv = []
            for file in uploaded_files:
                image = Image.open(file).convert("RGB")
                image_np = np.array(image)
                image_cv = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)
                images_cv.append(image_cv)

            # Apply Mertens Fusion
            merge_mertens = cv.createMergeMertens()
            fusion = merge_mertens.process(images_cv)

            # Normalize and convert to PIL
            fusion_8bit = np.clip(fusion * 255, 0, 255).astype('uint8')
            fusion_rgb = cv.cvtColor(fusion_8bit, cv.COLOR_BGR2RGB)
            fusion_pil = Image.fromarray(fusion_rgb)
            resized_fusion = resize_image_pil(fusion_pil)

            # Display result
            st.success("✅ HDR Image Processed Successfully!")
            st.markdown("### 🖼️ Input Images vs Output")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.image(images_preview[0], caption="Low", use_container_width=True)
            with col2:
                st.image(images_preview[1], caption="Mid", use_container_width=True)
            with col3:
                st.image(images_preview[2], caption="High", use_container_width=True)
            with col4:
                st.image(resized_fusion, caption="HDR Output", use_container_width=True)

            # Provide download button
            buffer = io.BytesIO()
            fusion_pil.save(buffer, format="JPEG")
            st.download_button(
                label="📥 Download HDR Image",
                data=buffer.getvalue(),
                file_name="hdr_output.jpg",
                mime="image/jpeg"
            )

        except Exception as e:
            st.error(f"❌ Error processing HDR image: {e}")
