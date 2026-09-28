import streamlit as st
import torch
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image


# ---------------------------------------------------------
# 1. Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Food Image Classification",
    page_icon="🍔",
    layout="centered"
)


# ---------------------------------------------------------
# 2. Load pre-trained MobileNetV3-Large
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    weights = MobileNet_V3_Large_Weights.DEFAULT

    model = mobilenet_v3_large(weights=weights)

    model.eval()

    categories = weights.meta["categories"]

    preprocess = weights.transforms()

    return model, categories, preprocess


model, categories, preprocess = load_model()


# ---------------------------------------------------------
# 3. Prediction function
# ---------------------------------------------------------

def predict_food(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Preprocess image
    input_image = preprocess(image)

    # Add batch dimension
    input_batch = input_image.unsqueeze(0)

    # Perform prediction
    with torch.no_grad():

        output = model(input_batch)

    # Convert output into probabilities
    probabilities = torch.nn.functional.softmax(
        output[0],
        dim=0
    )

    # Get top 5 predictions
    top5_prob, top5_indices = torch.topk(
        probabilities,
        5
    )

    predictions = []

    for probability, index in zip(
        top5_prob,
        top5_indices
    ):

        predictions.append({
            "class_name": categories[index.item()],
            "confidence": probability.item() * 100
        })

    # Get highest prediction
    predicted_index = top5_indices[0].item()

    predicted_class = categories[predicted_index]

    predicted_confidence = top5_prob[0].item() * 100

    return (
        predicted_class,
        predicted_confidence,
        predictions
    )


# ---------------------------------------------------------
# 4. Application title
# ---------------------------------------------------------

st.title("🍔 Food Image Classification")

st.write(
    "Upload a food image and classify it using "
    "a pre-trained MobileNetV3-Large model."
)


# ---------------------------------------------------------
# 5. Upload image
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a food image",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------------------------------------
# 6. Display image and prediction
# ---------------------------------------------------------

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file)

    # Display image
    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Selected Food Image",
        use_container_width=True
    )


    # Prediction button
    if st.button("Predict Food"):

        try:

            # Get predictions
            (
                predicted_class,
                predicted_confidence,
                predictions
            ) = predict_food(image)


            # -------------------------------------------------
            # Prediction result
            # -------------------------------------------------

            st.subheader("Prediction Result")

            st.success(
                f"Predicted Class: {predicted_class}"
            )

            st.write(
                f"Confidence: "
                f"{predicted_confidence:.2f}%"
            )


            # -------------------------------------------------
            # Top 5 predictions
            # -------------------------------------------------

            st.subheader("Top 5 Predictions")


            for i, prediction in enumerate(
                predictions,
                start=1
            ):

                st.write(
                    f"**{i}. {prediction['class_name']}** "
                    f"- {prediction['confidence']:.2f}%"
                )


                # Confidence bar
                st.progress(
                    min(
                        int(prediction["confidence"]),
                        100
                    )
                )


        except Exception as e:

            st.error(
                f"Could not process the image: {e}"
            )


# ---------------------------------------------------------
# 7. Information about the model
# ---------------------------------------------------------

st.markdown("---")

st.info(
    "This application uses the pre-trained "
    "MobileNetV3-Large model with ImageNet weights. "
    "No training is performed on the uploaded images."
)
