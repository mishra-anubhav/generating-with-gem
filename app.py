import streamlit as st
import os
from PIL import Image
import requests
from io import BytesIO
from dotenv import load_dotenv
import base64

# --- Core Modules ---
from google_ai_utils import initialize_vertex_ai
from describe_person_with_gemini import describe_person_with_gemini
from describe_garment_with_gemini import describe_garment_with_gemini
from final_prompt_builder import compose_final_prompt
from image_generation import generate_tryon_image_with_google_gen
from collage_generator import generate_collage, clean_background

# --- Search Functions ---
from upper_body import search_upper_body
from lower_body import search_lower_body
from shoes import search_shoes

# --- Init ---
load_dotenv()
try:
    initialize_vertex_ai()
except Exception as e:
    st.error(f"Vertex AI init failed: {e}")
    st.stop()

# --- Helpers ---
def standardized_image(url: str) -> str:
    return (
        '<div style="height:300px;width:300px;display:flex;justify-content:center;'
        'align-items:center;overflow:hidden;">'
        f'<img src="{url}" style="height:100%;width:auto;object-fit:contain;" />'
        '</div>'
    )

def url_to_pil(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, timeout=15, headers=headers)
        res.raise_for_status()
        return Image.open(BytesIO(res.content)).convert("RGB")
    except Exception as e:
        st.warning(f"Image fetch failed: {e}")
        return None

# --- Main ---
def main():
    st.set_page_config(page_title="Virtual Try-On", layout="centered")
    st.title("🧥 Virtual Try-On (Gemini + Imagen)")

    st.subheader("👫 Choose Gender")
    gender = st.radio("Gender", ["Male", "Female", "Unisex"], horizontal=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Upper Body")
        upper_kind = st.selectbox("Type", ["T-shirt", "Shirt", "Hoodie", "Sweater", "Jacket"])
        upper_kw = st.text_input("Keyword", key="upper_kw")
    with col2:
        st.header("Lower Body")
        lower_kind = st.selectbox("Type", ["Jeans", "Pants", "Shorts", "Skirt", "Trousers"])
        lower_kw = st.text_input("Keyword", key="lower_kw")
    with col3:
        st.header("Shoes")
        shoe_kind = st.selectbox("Type", ["Sneakers", "Boots", "Sandals", "Formal Shoes", "Running Shoes"])
        shoe_kw = st.text_input("Keyword", key="shoe_kw")

    ss = st.session_state
    ss.setdefault("upper_products", [])
    ss.setdefault("lower_products", [])
    ss.setdefault("shoe_products", [])
    ss.setdefault("idx", 0)

    ss.setdefault("person_image_pil", None)
    ss.setdefault("upper_image_pil", None)
    ss.setdefault("lower_image_pil", None)
    ss.setdefault("shoes_image_pil", None)

    ss.setdefault("user_photo_bytes", None)
    ss.setdefault("person_description", "")
    ss.setdefault("garment_descriptions", {})
    ss.setdefault("final_tryon_prompt", "")
    ss.setdefault("generated_image_data_url", None)
    ss.setdefault("person_img", None)

    if st.button("🔍 Search Outfits"):
        with st.spinner("Searching outfits..."):
            ss.upper_products = search_upper_body(upper_kind, gender, upper_kw, 5)
            ss.lower_products = search_lower_body(lower_kind, gender, lower_kw, 5)
            ss.shoe_products = search_shoes(shoe_kind, gender, shoe_kw, 5)
            ss.idx = 0
            ss.upper_image_pil = None
            ss.lower_image_pil = None
            ss.shoes_image_pil = None
            ss.person_description = ""
            ss.garment_descriptions = {}
            ss.final_tryon_prompt = ""
            ss.generated_image_data_url = None
            ss.person_img = None

    if ss.upper_products or ss.lower_products or ss.shoe_products:
        st.subheader("✨ Selected Outfit")
        cols = st.columns(3)
        garment_sources = [
            ("Upper", ss.upper_products, upper_kind, "upper_image_pil", "upper"),
            ("Lower", ss.lower_products, lower_kind, "lower_image_pil", "lower"),
            ("Shoes", ss.shoe_products, shoe_kind, "shoes_image_pil", "shoes"),
        ]

        for i, (label, products, kind, session_key, file_prefix) in enumerate(garment_sources):
            with cols[i]:
                if products and ss.idx < len(products):
                    item = products[ss.idx]
                    st.markdown(standardized_image(item["thumbnail"]), unsafe_allow_html=True)
                    st.caption(item["title"])
                    if st.button(f"✅ Use this {label.lower()}", key=f"btn_use_{label.lower()}"):
                        img = url_to_pil(item["thumbnail"])
                        if img:
                            img_clean = clean_background(img)
                            ss[session_key] = img_clean
                            os.makedirs("input", exist_ok=True)
                            img_clean.save(f"input/{file_prefix}.png")
                            st.success(f"Cleaned and saved {label} image.")

        if any(len(p) > 1 for p in [ss.upper_products, ss.lower_products, ss.shoe_products]):
            col_prev, col_next = st.columns([1, 1])
            with col_prev:
                if st.button("⬅️ Prev"):
                    ss.idx = max(0, ss.idx - 1)
            with col_next:
                if st.button("➡️ Next"):
                    max_len = max(len(ss.upper_products), len(ss.lower_products), len(ss.shoe_products))
                    ss.idx = min(max_len - 1, ss.idx + 1)

        st.subheader("📸 Upload Your Photo")
        uploaded_photo = st.file_uploader("Upload full-body image", type=["png", "jpg", "jpeg"])
        if uploaded_photo:
            if ss.user_photo_bytes is None or uploaded_photo.getvalue() != ss.user_photo_bytes:
                ss.user_photo_bytes = uploaded_photo.getvalue()
                try:
                    img = Image.open(BytesIO(ss.user_photo_bytes)).convert("RGB")
                    img_clean = clean_background(img)
                    ss.person_image_pil = img_clean
                    os.makedirs("input", exist_ok=True)
                    img_clean.save("input/person.png")
                    st.success("Cleaned and saved person photo.")
                    ss.person_description = ""
                    ss.final_tryon_prompt = ""
                    ss.generated_image_data_url = None
                except Exception as e:
                    st.error(f"Image error: {e}")
                    ss.person_image_pil = None

        if ss.person_image_pil:
            st.image(ss.person_image_pil, caption="👤 Your Uploaded Photo", use_column_width=True)

        if ss.person_image_pil and (ss.upper_image_pil or ss.lower_image_pil or ss.shoes_image_pil) and not ss.person_description:
            if st.button("📜 Describe Images with Gemini"):
                with st.spinner("Describing..."):
                    person_image = Image.open("input/person.png")
                    ss.person_description = describe_person_with_gemini(person_image)
                    st.write(f"👤 Person: {ss.person_description[:300]}...")

                    ss.garment_descriptions = {}
                    if ss.upper_image_pil:
                        desc = describe_garment_with_gemini(Image.open("input/upper.png"), upper_kind, ss.upper_products[ss.idx]['title'])
                        ss.garment_descriptions["upper"] = desc
                        st.write(f"👕 {upper_kind}: {desc[:300]}...")
                    if ss.lower_image_pil:
                        desc = describe_garment_with_gemini(Image.open("input/lower.png"), lower_kind, ss.lower_products[ss.idx]['title'])
                        ss.garment_descriptions["lower"] = desc
                        st.write(f"👖 {lower_kind}: {desc[:300]}...")
                    if ss.shoes_image_pil:
                        desc = describe_garment_with_gemini(Image.open("input/shoes.png"), shoe_kind, ss.shoe_products[ss.idx]['title'])
                        ss.garment_descriptions["shoes"] = desc
                        st.write(f"👟 {shoe_kind}: {desc[:300]}...")

        if ss.person_image_pil and any([ss.upper_image_pil, ss.lower_image_pil, ss.shoes_image_pil]):
            with st.spinner("🥩 Creating collage with garments and person..."):
                collage = generate_collage(
                    Image.open("input/person.png"),
                    Image.open("input/shoes.png") if os.path.exists("input/shoes.png") else None,
                    Image.open("input/lower.png") if os.path.exists("input/lower.png") else None,
                    Image.open("input/upper.png") if os.path.exists("input/upper.png") else None,
                    
                )
                ss.person_img = collage
                os.makedirs("input", exist_ok=True)
                collage.save("input/reference_collage.png")
                st.image(collage, caption="🥩 Reference Collage", use_column_width=True)

        if st.button("✨ Compose Final Prompt"):
            if ss.person_description and ss.garment_descriptions:
                with st.spinner("Building prompt..."):
                    ss.final_tryon_prompt = compose_final_prompt(ss.person_description, ss.garment_descriptions)
                    st.text_area("Prompt", ss.final_tryon_prompt, height=200)
            else:
                st.warning("Missing person or garment descriptions.")
        if ss.garment_descriptions:
            st.subheader("🧵 Garment Descriptions")
            for k, desc in ss.garment_descriptions.items():
                st.markdown(f"**{k.title()}**: {desc[:400]}...")

        if ss.final_tryon_prompt and ss.person_img and not ss.generated_image_data_url:
            if st.button("🚀 Generate Try-On Image"):
                with st.spinner("Calling Imagen 3..."):
                    try:
                        try:
                            collage_image = Image.open("input/reference_collage.png").convert("RGB")
                            print(">> Final prompt preview:\n", ss.final_tryon_prompt[:500])
                            
                            result = generate_tryon_image_with_google_gen(
                                person_image=collage_image,
                                final_prompt=ss.final_tryon_prompt
                            )
                            print(">> Result:", str(result)[:200])  # Just preview beginning

                            if result and result.startswith("data:image"):
                                ss.generated_image_data_url = result
                                st.success("✅ Image generated and saved.")
                            else:
                                st.error("❌ Imagen 3 API returned no valid image. Check prompt or image formatting.")

                        except Exception as e:
                            st.error(f"🔥 Error while generating image: {e}")


                        if ss.generated_image_data_url:
                            base64_data = ss.generated_image_data_url.split(",")[1]
                            output_bytes = base64.b64decode(base64_data)
                            os.makedirs("output", exist_ok=True)
                            with open("output/generated_tryon.png", "wb") as f:
                                f.write(output_bytes)
                            st.success("✅ Image generated and saved to output folder.")
                        else:
                            st.error("❌ Imagen 3 API returned no image.")
                    except Exception as e:
                        st.error(f"🔥 Error during image generation: {e}")

        if ss.generated_image_data_url:
            st.subheader("👗 Try-On Result")
            st.image(ss.generated_image_data_url, use_column_width=True)

if __name__ == "__main__":
    main()