from flask import Flask, request, render_template_string
import torch
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
import io

# ---------------------------------------------------------
# 1. Create Flask application
# ---------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------
# 2. Load the pre-trained MobileNetV3-Large model
# ---------------------------------------------------------
weights = MobileNet_V3_Large_Weights.DEFAULT
model = mobilenet_v3_large(weights=weights)
model.eval()

# Load ImageNet class names
categories = weights.meta["categories"]

# Get the official preprocessing transformations
preprocess = weights.transforms()


# ---------------------------------------------------------
# 3. Prediction function
# ---------------------------------------------------------
def predict_food(image):
    """
    Takes a PIL image as input and returns:
    - predicted class
    - confidence
    - top 5 predictions
    """

    # Convert image to RGB
    image = image.convert("RGB")

    # Preprocess the image
    input_image = preprocess(image)

    # Add batch dimension
    input_batch = input_image.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Convert raw scores into probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)

    predictions = []

    for probability, index in zip(top5_prob, top5_indices):
        predictions.append({
            "class_name": categories[index.item()],
            "confidence": f"{probability.item() * 100:.2f}%"
        })

    # Highest probability prediction
    predicted_index = top5_indices[0].item()

    predicted_class = categories[predicted_index]
    predicted_confidence = top5_prob[0].item() * 100

    return predicted_class, predicted_confidence, predictions


# ---------------------------------------------------------
# 4. HTML interface
# ---------------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Food Image Classification</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f4f4;
            text-align: center;
            margin: 0;
            padding: 40px;
        }

        .container {
            max-width: 700px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.10);
        }

        h1 {
            margin-bottom: 10px;
        }

        .subtitle {
            color: #555;
            margin-bottom: 25px;
        }

        input[type="file"] {
            margin: 15px;
        }

        button {
            padding: 10px 22px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            background: #333;
            color: white;
            font-size: 16px;
        }

        button:hover {
            background: #555;
        }

        img {
            max-width: 450px;
            max-height: 350px;
            margin-top: 25px;
            border-radius: 8px;
        }

        .result {
            margin-top: 25px;
            padding: 20px;
            background: #f0f0f0;
            border-radius: 8px;
        }

        .prediction {
            font-size: 24px;
            font-weight: bold;
        }

        table {
            width: 100%;
            margin-top: 20px;
            border-collapse: collapse;
        }

        th, td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }

        .note {
            margin-top: 20px;
            font-size: 14px;
            color: #666;
        }

        .error {
            color: #b00020;
            margin-top: 20px;
        }
    </style>
</head>

<body>

<div class="container">

    <h1>Food Image Classification</h1>

    <p class="subtitle">
        Pre-trained MobileNetV3-Large
    </p>

    <form method="POST" enctype="multipart/form-data">

        <input
            type="file"
            name="image"
            accept="image/*"
            required
        >

        <br>

        <button type="submit">
            Predict Food
        </button>

    </form>


    {% if error %}

        <p class="error">{{ error }}</p>

    {% endif %}


    {% if image_data %}

        <img
            src="data:image/jpeg;base64,{{ image_data }}"
            alt="Uploaded Image"
        >

    {% endif %}


    {% if predicted_class %}

        <div class="result">

            <h2>Prediction Result</h2>

            <p class="prediction">
                {{ predicted_class }}
            </p>

            <p>
                Confidence:
                <strong>{{ "%.2f"|format(predicted_confidence) }}%</strong>
            </p>

            <h3>Top 5 Predictions</h3>

            <table>

                <tr>
                    <th>Class</th>
                    <th>Confidence</th>
                </tr>

                {% for prediction in predictions %}

                <tr>
                    <td>{{ prediction.class_name }}</td>
                    <td>{{ prediction.confidence }}</td>
                </tr>

                {% endfor %}

            </table>

        </div>

    {% endif %}


    <p class="note">
        Note: The MobileNetV3-Large ImageNet weights are a general
        image-classification model, not a food-specific model.
        Therefore, some food images may be assigned to visually
        related ImageNet classes.
    </p>

</div>

</body>
</html>
"""


# ---------------------------------------------------------
# 5. Home page and prediction route
# ---------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    predicted_class = None
    predicted_confidence = None
    predictions = []
    image_data = None
    error = None

    if request.method == "POST":

        # Check whether an image was uploaded
        if "image" not in request.files:
            error = "Please upload an image."
            return render_template_string(
                HTML_PAGE,
                predicted_class=predicted_class,
                predicted_confidence=predicted_confidence,
                predictions=predictions,
                image_data=image_data,
                error=error
            )

        file = request.files["image"]

        if file.filename == "":
            error = "Please select an image."
            return render_template_string(
                HTML_PAGE,
                predicted_class=predicted_class,
                predicted_confidence=predicted_confidence,
                predictions=predictions,
                image_data=image_data,
                error=error
            )

        try:
            # Read uploaded image
            image_bytes = file.read()

            # Open image with Pillow
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Generate prediction
            (
                predicted_class,
                predicted_confidence,
                predictions
            ) = predict_food(image)

            # Convert image to base64 so it can be displayed
            import base64

            image_data = base64.b64encode(
                image_bytes
            ).decode("utf-8")

        except Exception as e:
            error = f"Could not process the image: {e}"

    return render_template_string(
        HTML_PAGE,
        predicted_class=predicted_class,
        predicted_confidence=predicted_confidence,
        predictions=predictions,
        image_data=image_data,
        error=error
    )


# ---------------------------------------------------------
# 6. Run the application
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Starting Food Image Classification application...")
    print("Open the displayed local address in your browser.")
    app.run(debug=True)
