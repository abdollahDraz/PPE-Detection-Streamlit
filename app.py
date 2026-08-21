import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2




st.set_page_config(
    page_title="PPE SAFETY AI",
    page_icon="🛡️",
    layout="wide"
)




model = YOLO("PPE _Complex_ Model .pt")




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




uploaded_file = st.file_uploader(
    "📤 Upload Worker Image",
    type=["jpg", "jpeg", "png"],
    help="Upload an image of a worker to check PPE compliance."
)



if uploaded_file is not None:

    st.success("✅ Uploaded Image Successfully")



    img = Image.open(uploaded_file)




    if st.session_state.get("uploaded_filename") != uploaded_file.name:

        st.session_state.res = None

        st.session_state.prediction_done = False

        for ppe in st.session_state.required_ppe:
            st.session_state.required_ppe[ppe] = False

        st.session_state.uploaded_filename = uploaded_file.name



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




        st.session_state.res = result.plot()

        st.session_state.prediction_done = True




        for cls_id in result.boxes.cls:

            cls_id = int(cls_id)

            cls = result.names[cls_id]




            if cls in st.session_state.required_ppe:

                st.session_state.required_ppe[cls] = True


        st.success("✅ Prediction completed successfully!")




if uploaded_file is not None:

    st.divider()


    col1, col2 = st.columns(2)




    with col1:

        st.info("🖼️ Original Image")

        st.image(
            img,
            use_container_width=True
        )




    with col2:

        st.info("🔍 Predicted Image")

        if st.session_state.res is not None:

            st.image(
                st.session_state.res,
                channels="BGR",
                use_container_width=True
            )

        else:

            st.warning(
                "Click 'Check up your image' to run prediction."
            )



