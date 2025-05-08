# image_generation.py

import os
import io
import base64
from PIL import Image
from google.cloud.aiplatform_v1.services import prediction_service

# Model configuration
MODEL_ID = "imagen-3.0-generate-002"

def generate_tryon_image_with_google_gen(person_image: Image.Image, final_prompt: str) -> str:
    """
    Calls the Google Cloud Imagen 3 API to generate a try-on image using a reference collage.

    Args:
        person_image: A PIL Image object (usually a collage with garments and person).
        final_prompt: A detailed natural language prompt referencing 'person [1]'.

    Returns:
        A base64 data URL of the generated image, or None on failure.
    """
    # Step 1: Load credentials
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")

    if not project_id or not location:
        raise ValueError("❌ GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION not set in environment.")

    # Step 2: Encode the collage image to base64 (as PNG to preserve transparency)
    try:
        img_byte_arr = io.BytesIO()
        person_image.save(img_byte_arr, format="PNG")  # Collage with transparency
        person_image_bytes = img_byte_arr.getvalue()
        person_image_base64 = base64.b64encode(person_image_bytes).decode("utf-8")
        mime_type = "image/png"
    except Exception as e:
        print(f"❌ Error encoding reference image: {e}")
        return None

    # Step 3: Build the payload
    instances = [
        {
            "prompt": final_prompt,
            "reference_images": [
                {
                    "image": {
                        "bytesBase64Encoded": person_image_base64
                    },
                    "subject_type": "SUBJECT_TYPE_PERSON",
                    "reference_id": "1"
                }
            ],
            "sampleCount": 1,
            "aspect_ratio": "1:1"
        }
    ]

    # Step 4: Setup Prediction Client
    client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    client = prediction_service.PredictionServiceClient(client_options=client_options)
    endpoint = f"projects/{project_id}/locations/{location}/publishers/google/models/{MODEL_ID}"

    # Step 5: Make the API call
    try:
        response = client.predict(endpoint=endpoint, instances=instances)
        if response and response.predictions:
            prediction_value = response.predictions[0]
            try:
                prediction_dict = dict(prediction_value)
            except Exception as e:
                print(f"❌ Failed to parse prediction: {e}")
                return None

            if "bytesBase64Encoded" in prediction_dict and "mimeType" in prediction_dict:
                base64_image = prediction_dict["bytesBase64Encoded"]
                mime_type = prediction_dict["mimeType"]
                return f"data:{mime_type};base64,{base64_image}"
            else:
                print("⚠️ Warning: Missing 'bytesBase64Encoded' or 'mimeType' in response.")
                return None
        else:
            print("⚠️ Warning: Empty prediction response.")
            return None

    except Exception as e:
        print(f"❌ Error during prediction call: {e}")
        return None
