from PIL import Image, ImageDraw, ImageFont
import io
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager

def get_font(size=60):
    """
    Loads a font using matplotlib's font manager to avoid font_roboto issues.
    Tries to load Roboto, falls back to DejaVu Sans or system default.
    """
    try:
        # Try Roboto (if manually installed)
        roboto_candidates = [f for f in font_manager.findSystemFonts() if "Roboto-Regular" in f]
        if roboto_candidates:
            return ImageFont.truetype(roboto_candidates[0], size)
        else:
            # Fall back to DejaVu Sans (bundled with matplotlib)
            dejavu = font_manager.findfont("DejaVu Sans")
            return ImageFont.truetype(dejavu, size)
    except Exception as e:
        print(f"⚠️ Font fallback error: {e}")
        return ImageFont.load_default()

# Optional: Use rembg for background removal if available
try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    print("WARNING: rembg not available — backgrounds will not be removed.")
    REMBG_AVAILABLE = False


def clean_background(img: Image.Image) -> Image.Image:
    """
    Removes background from a PIL image using rembg if available.
    """
    if REMBG_AVAILABLE:
        try:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            no_bg_bytes = rembg_remove(img_byte_arr)
            return Image.open(io.BytesIO(no_bg_bytes)).convert("RGBA")
        except Exception as e:
            print(f"Background removal failed: {e}")
    return img.convert("RGBA")

def generate_collage(person: Image.Image, shoes: Image.Image, lower: Image.Image, upper: Image.Image, width=512, padding=20) -> Image.Image:
    """
    Arranges collage in this order:
    1. Person | 2. Shoes | 3. Lower Body | 4. Upper Body
    Transparent background, with vertical dividers and large Roboto labels.
    """

    # Step 1: Clean backgrounds
    person = clean_background(person)
    shoes = clean_background(shoes)
    lower = clean_background(lower)
    upper = clean_background(upper)

    # Step 2: Resize to uniform height
    target_height = 512
    items = [person, shoes, lower, upper]
    labels = ["1. Person", "2. Shoes", "3. Lower Body", "4. Upper Body"]
    resized = []

    for img in items:
        aspect_ratio = img.width / img.height
        new_width = int(target_height * aspect_ratio)
        resized.append(img.resize((new_width, target_height)))

    # Step 3: Load Roboto font
    font = get_font(size=60)
    label_height = 100  # Enough space for large font

    total_width = sum(img.width for img in resized) + padding * (len(resized) + 1)
    total_height = target_height + label_height + padding * 2

    # Step 4: Create transparent canvas
    collage = Image.new("RGBA", (total_width, total_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(collage)

    # Step 5: Paste images + labels + vertical lines
    x = padding
    for i, img in enumerate(resized):
        bbox = draw.textbbox((0, 0), labels[i], font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (img.width - text_width) // 2

        # Draw label
        draw.text((text_x, padding), labels[i], font=font, fill=(0, 0, 0, 255))

        # Paste image
        collage.paste(img, (x, padding + label_height), mask=img)

        # Draw vertical divider (except after last image)
        if i < len(resized) - 1:
            line_x = x + img.width + padding // 2
            draw.line([(line_x, 0), (line_x, total_height)], fill=(0, 0, 0, 64), width=2)

        x += img.width + padding

    return collage
