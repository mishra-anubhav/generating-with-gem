# person_processing.py
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from PIL import Image
import io

# Make sure to call initialize_vertex_ai() in your main app.py or a setup file
# before calling functions in this module.

def describe_person_with_gemini(person_image: Image.Image) -> str:
    """
    Uses Gemini to describe the person's appearance and pose from a PIL Image.

    Args:
        person_image: A PIL Image object of the person.

    Returns:
        A string description of the person, or an error message.
    """
    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        # Use a standard format like PNG or JPEG for the API
        person_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Create a Part object from the image bytes
        # Ensure mime_type matches the format used in save()
        image_part = Part.from_data(data=img_byte_arr, mime_type='image/png')

        # Create the text prompt for describing the person
        # Be specific about what details are needed for redressing
        prompt_text = (
            "Describe the person in this image in detail, focusing on their physical attributes, "
            "body shape, pose, stance, skin tone, hair style and color, and facial features. "
            "Ignore any clothing they are currently wearing. "
            "This description is for an AI model that will generate a new image of this person "
            "wearing different clothes, so provide details that help preserve their identity and posture. "
            "Provide a concise summary description. Return only the description text."
        )

        # Get the Gemini model instance (use the one that worked in your test)
        # 'gemini-1.0-pro-001' is often a good choice for multimodal tasks
        model = GenerativeModel("gemini-2.5-pro-preview-03-25") # <-- Use the exact model name that worked

        # Send the prompt containing text and the image part to Gemini
        print("DEBUG: Sending person image to Gemini for description...")
        response = model.generate_content([prompt_text, image_part])
        print("DEBUG: Received response from Gemini.")

        # Return the text response from Gemini
        if response and response.text:
             return response.text.strip()
        else:
             print("DEBUG: Gemini response did not contain text.")
             return "Could not generate person description."

    except Exception as e:
        print(f"Error describing person with Gemini: {e}")
        # You might want more sophisticated error handling here
        return f"Error describing person: {e}"

# Example of how you might use this function:
# from PIL import Image
# from person_processing import describe_person_with_gemini
# from google_ai_utils import initialize_vertex_ai # Assuming initialization is here or in app.py

# # Assuming initialization is handled elsewhere before this point
# # initialize_vertex_ai()

# # Load a sample person image (replace with actual image loading)
# try:
#     sample_person_image = Image.open("path/to/your/sample/person.jpg")
#     description = describe_person_with_gemini(sample_person_image)
#     print(f"Generated Description: {description}")
# except FileNotFoundError:
#     print("Sample image not found.")
# except Exception as e:
#      print(f"An error occurred: {e}")