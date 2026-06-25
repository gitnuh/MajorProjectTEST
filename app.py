import torch
import streamlit as st

from PIL import Image
from torchvision.models import ResNet18_Weights

from models.feature_extractor import (
    load_feature_extractor,
    extract_features,
)

from detector.ood_detector import (
    OODDetector
)


# --------------------------------------------------
# CIFAR-10 class labels
# --------------------------------------------------

CIFAR_CLASSES = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Mahalanobis OOD Detector",
    layout="centered"
)


# --------------------------------------------------
# Load model and detector only once
# --------------------------------------------------

@st.cache_resource
def load_resources():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = (
        load_feature_extractor()
        .to(device)
    )

    detector = OODDetector()

    return model, detector, device


# --------------------------------------------------
# Convert uploaded image to ResNet format
# --------------------------------------------------

def preprocess_image(image):

    transform = (
        ResNet18_Weights
        .DEFAULT
        .transforms()
    )

    if image.mode != "RGB":
        image = image.convert("RGB")

    image_tensor = transform(image)

    # Add batch dimension
    return image_tensor.unsqueeze(0)


# --------------------------------------------------
# Main UI
# --------------------------------------------------

def main():

    st.title(
        "Think of title like 'titli' :P"
    )

    st.markdown(
        """
        DetectING whether an uploaded image belongs to the
        CIFAR-10 distribution using Mahalanobis Distance. 
        """
    )

    st.divider()

    model, detector, device = (
        load_resources()
    )

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        return

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Analyze Image"):

        with st.spinner(
            "Running Inference..."
        ):

            # -------------------------
            # Image → Feature Vector
            # -------------------------

            input_tensor = (
                preprocess_image(image)
            )

            features = (
                extract_features(
                    model,
                    input_tensor,
                    device
                )
            )

            # First image in batch
            feature_vector = (
                features[0]
                .cpu()
            )

            # -------------------------
            # OOD Detection
            # -------------------------

            result = detector.detect(
                feature_vector
            )

            class_id = result["class_id"]

            distance = result["distance"]

            is_ood = result["is_ood"]

            # -------------------------
            # FAISS Detection
            # -------------------------

            faiss_result = detector.faiss_detect(
                feature_vector
            )

            faiss_distance = (
                faiss_result["distance"]
            )

            faiss_ood = (
                faiss_result["is_ood"]
            )   
#------------------------------------------------ faiss nuha
            predicted_class = (
                CIFAR_CLASSES[class_id]
            )

            # -------------------------
            # Get all class distances
            # -------------------------

            distances = (
                detector
                .scorer
                .distance(feature_vector)
            )

            # Pair each class with distance
            class_distance_pairs = list(
                zip(
                    CIFAR_CLASSES,
                    distances
                )
            )

            # Smaller distance = closer class
            class_distance_pairs.sort(
                key=lambda x: x[1]
            )

            # -------------------------
            # Prediction Summary
            # -------------------------

            st.divider()

            st.subheader(
                "Prediction Summary"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted Class",
                    predicted_class
                )

                st.metric(
                    "Distance",
                    f"{distance:.2f}"
                )
#---------------------------------------------nua
                st.metric(
                    "FAISS Distance",
                    f"{faiss_distance:.2f}"
                )
#----------------------------------------------
            with col2:

                st.metric(
                    "Threshold",
                    f"{detector.threshold:.2f}"
                )

                if is_ood:

                    st.error(
                        "OOD Detected"
                    )

                else:

                    st.success(
                        "In Distribution"
                    )
#----------------------------------------------nua                    
                if faiss_ood:

                    st.warning(
                        "FAISS: OOD"
                    )

                else:

                    st.info(
                        "FAISS: In Distribution"
                    )
#---------------------------------------------------

            # -------------------------
            # Top 3 Closest Classes
            # -------------------------

            st.divider()

            st.subheader(
                "Top 3 Closest Classes"
            )

            for rank, (
                class_name,
                dist
            ) in enumerate(
                class_distance_pairs[:3],
                start=1
            ):

                st.write(
                    f"{rank}. **{class_name}** — {dist:.2f}"
                )

            # -------------------------
            # Technical Details
            # -------------------------

            with st.expander(
                "Technical Details"
            ):

                st.write(
                    f"Device: {device}"
                )

                st.write(
                    "Feature Extractor: ResNet18"
                )

                st.write(
                    "Feature Dimension: 512"
                )

                st.write(
                    f"OOD Threshold: {detector.threshold:.2f}"
                )

                st.write(
                    f"Feature Shape: {feature_vector.shape}"
                )

                st.write(
                    """
                    Workflow:

                    Image
                    → ResNet18
                    → Feature Vector
                    → Mahalanobis Distance
                    → Threshold Check
                    → ID / OOD Decision
                    """
                )


if __name__ == "__main__":
    main()