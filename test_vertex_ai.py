# Option B (Better Test for your use case): A basic multimodal call sending a sample image and asking Gemini to describe it
from vertexai.generative_models import GenerativeModel, Part
from PIL import Image
import io
import requests # Or load a local image
import base64 # Import base64

# Ensure vertexai is initialized (assuming GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION are set via .env or environment)
import vertexai
import os
from dotenv import load_dotenv

load_dotenv() # Make sure this is called!
print(f"DEBUG: GOOGLE_CLOUD_PROJECT={os.getenv('GOOGLE_CLOUD_PROJECT')}")
print(f"DEBUG: GOOGLE_CLOUD_LOCATION={os.getenv('GOOGLE_CLOUD_LOCATION')}")
print(f"DEBUG: GOOGLE_APPLICATION_CREDENTIALS={os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")

try:
    vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location=os.getenv("GOOGLE_CLOUD_LOCATION"))
    print("Vertex AI initialized.")
except Exception as e:
    print(f"Failed to initialize Vertex AI: {e}")
    exit() # Exit if initialization fails

try:
    # 2. Load the image into a PIL Image object
    image_path ="/Users/anubhavmishra/Desktop/Clothes-finder/Image-generation with google/sample/sample.png"
    img = Image.open(image_path)

    # 3. Convert the PIL Image to bytes in a standard format (e.g., PNG)
    img_byte_arr = io.BytesIO()
    # Use a standard format string supported by PIL.Image.save
    # PNG is generally recommended for image data to avoid re-compression loss
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # 4. Create a Part object *explicitly* from the image bytes and mime type
    # Ensure the mime_type matches the format you saved as
    image_part = Part.from_data(data=img_byte_arr, mime_type='image/png') # <--- Use from_data here

    # 5. Create the text prompt
    prompt_text = "Describe what you see in this image."

    # 6. Combine text and image parts for the multimodal prompt
    contents = [prompt_text, image_part]

    # 7. Call the Gemini API
    model = GenerativeModel("gemini-2.0-flash-001") # Or "gemini-1.0-pro"
    print("DEBUG: Sending multimodal request to Gemini...")
    response = model.generate_content(contents) # Error happened here before

    # 8. Print the response
    print("Response from Gemini:")
    print(response.text) # This accesses the text part of the response

except Exception as e:
    print(f"Test failed: {e}")

print("Multimodal test finished.")