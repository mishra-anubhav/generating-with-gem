from vertexai.preview.generative_models import GenerativeModel

DETAILED_PROMPT_TEMPLATE = """
Generate a realistic, full-body image of a person wearing the described outfit. 
Use the reference image to preserve the person's identity, skin tone, body shape, and pose exactly.

Important: The input collage contains 4 images in this order:
1. Person reference – use for face-identity, skin tone, pose, and body shape.
2. Shoes – use for rendering footwear.
3. Lower-body garment – use for pants, jeans, trousers.
4. Upper-body garment – use for shirts, jackets, or tops.

Garment descriptions:
Upper body clothing:
{upper_desc}

Lower body clothing:
{lower_desc}

Shoes:
{shoe_desc}

Instructions:
- Do not alter the person's face, hair, skin tone, posture, or proportions.
- Face should always be present
- Match the exact color, material, fit, and detailing of each garment.
- Render clothing naturally on the body.
- Use a clean studio background with soft, even lighting.
- Do not add accessories unless described.
"""

def summarize_with_gemini(prompt: str) -> str:
    model = GenerativeModel("gemini-2.5-pro-preview-03-25")
    chat = model.start_chat()

    full_prompt = (
        "You are a prompt optimizer for image generation models like Imagen. "
        "Rewrite the following prompt to be under 800 characters. "
        "Use clean, direct phrasing. Emphasize the reference image order and instructions for each garment placement. "
        "Do not remove information about garment details or identity preservation.\n\n"
        f"{prompt}"
    )

    response = chat.send_message(full_prompt)
    return response.text.strip()

def compose_final_prompt(person_desc: str, garment_descs: dict, viewpoint: str = "from the front") -> str:
    upper_desc = garment_descs.get("upper", "Not specified.")
    lower_desc = garment_descs.get("lower", "Not specified.")
    shoe_desc = garment_descs.get("shoes", "Not specified.")

    detailed_prompt = DETAILED_PROMPT_TEMPLATE.format(
        person_description=person_desc or "No person description provided.",
        upper_desc=upper_desc,
        lower_desc=lower_desc,
        shoe_desc=shoe_desc,
        viewpoint=viewpoint,
    ).strip()

    summarized_prompt = summarize_with_gemini(detailed_prompt)
    return summarized_prompt
