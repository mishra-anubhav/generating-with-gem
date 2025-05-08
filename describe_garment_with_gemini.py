# garment_processing.py
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from PIL import Image
import io

def describe_garment_with_gemini(garment_image: Image.Image, garment_type: str, garment_title: str = "") -> str:
    """
    Uses Gemini to describe a garment image, enhanced by its type and product name/title.

    Args:
        garment_image: A PIL Image object of the garment.
        garment_type: The type of garment (e.g., "T-shirt", "Jeans", "Sneakers").
        garment_title: (Optional) The product title from SerpAPI to give Gemini more brand/style info.

    Returns:
        A string description of the garment.
    """
    try:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        garment_image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        image_part = Part.from_data(data=image_bytes, mime_type='image/png')

        # Format the text prompt based on garment type
        garment_title_part = f"The product is called '{garment_title}'. " if garment_title else ""

        prompt_text = (
            f"{garment_title_part}"
            f"This is a {garment_type}. Describe only this {garment_type}, focusing on:\n"
            f"- Color (primary + secondary tones)\n"
            f"- Texture and fabric (e.g., suede, cotton, denim, etc.)\n"
            f"- Shape, silhouette, fit (e.g., slim fit, cropped, high-waisted)\n"
            f"- Fine details: patterns, logos, laces, zippers, collar, stitching, sole (for shoes)\n"
            f"- Brand or style hints if visible\n\n"
            f"Output a full, catalog-style description that helps an AI model recreate it exactly. "
            f"Do NOT include descriptions of background or mannequin. Only return description text."
        )

        model = GenerativeModel("gemini-2.5-pro-preview-03-25")
        response = model.generate_content([prompt_text, image_part])

        if response and response.text:
            return response.text.strip()
        else:
            return f"Could not generate description for {garment_type}."

    except Exception as e:
        return f"Error describing {garment_type}: {e}"
