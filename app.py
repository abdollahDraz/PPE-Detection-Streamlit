import streamlit as st
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import cv2



# Page Configuration


st.set_page_config(
    page_title="PPE SAFETY AI",
    page_icon="🛡️",
    layout="wide"
)



# Load YOLO Model


model_path = hf_hub_download(
    repo_id="abdollah111/ppe-detection-model",
    filename="PPE _Complex_ Model .pt"
)

model = YOLO(model_path)



# Session State


if "required_ppe" not in st.session_state:
    st.session_state.required_ppe = {
        "helmet": False,
        "vest": False,
        "gloves": False,
        "boots": False
    }


if "res" not in st.session_state:
    st.session_state.res = None


if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False


if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None



# Title


st.title("AI-Powered Personal Protective Equipment Detection")


st.markdown("""
### 🛡️ Smart Safety Compliance Detection

This AI-powered system uses **YOLO object detection** to analyze workers
and verify that all required Personal Protective Equipment (PPE) is worn.

The system checks for:

- 🪖 Helmet
- 🦺 Safety Vest
- 🧤 Gloves
- 🥾 Safety Boots
- 👤 Human

After analyzing the image, the system determines whether the worker is
**SAFE** and fully equipped with the required PPE, or **NOT SAFE** if any
required safety equipment is missing.

**Goal:** Improve workplace safety through automated PPE compliance detection.
""")


st.divider()


# Upload Image


uploaded_file = st.file_uploader(
    "📤 Upload Worker Image",
    type=["jpg", "jpeg", "png"],
    help="Upload an image of a worker to check PPE compliance."
)


if uploaded_file is not None:

    st.success("✅ Uploaded Image Successfully")

    img = Image.open(uploaded_file)


    
    # Reset State for New Image


    if st.session_state.uploaded_filename != uploaded_file.name:

        st.session_state.res = None
        st.session_state.prediction_done = False

        for ppe in st.session_state.required_ppe:
            st.session_state.required_ppe[ppe] = False

        st.session_state.uploaded_filename = uploaded_file.name


   
    # Prediction


    if st.button("🔍 Check up your image"):

        for ppe in st.session_state.required_ppe:
            st.session_state.required_ppe[ppe] = False


        image_bytes = uploaded_file.getvalue()

        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )


        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        result = model.predict(
            source=image,
            conf=0.25,
            save=False
        )[0]


      
        # Draw Predictions
       

        annotated_image = result.plot()

        annotated_image = cv2.cvtColor(
            annotated_image,
            cv2.COLOR_BGR2RGB
        )

        st.session_state.res = annotated_image

        st.session_state.prediction_done = True


        # Check PPE Classes
     

        for cls_id in result.boxes.cls:

            cls_id = int(cls_id)

            cls = result.names[cls_id]


            if cls in st.session_state.required_ppe:

                st.session_state.required_ppe[cls] = True


        st.success("✅ Prediction completed successfully!")


# Display Results


if uploaded_file is not None:

    st.divider()

    col1, col2 = st.columns(2)


    # Original Image
  

    with col1:

        st.info("🖼️ Original Image")

        st.image(
            img,
            use_container_width=True
        )



    # Predicted Image
   

    with col2:

        st.info("🔍 Predicted Image")


        if st.session_state.res is not None:

            st.image(
                st.session_state.res,
                use_container_width=True
            )

        else:

            st.warning(
                "Click 'Check up your image' to run prediction."
            )
