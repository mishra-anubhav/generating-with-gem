# google_ai_utils.py
import os
import vertexai
from dotenv import load_dotenv

def initialize_vertex_ai():
    """Initializes the Vertex AI SDK using environment variables."""
    load_dotenv() # Load variables from .env file

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") # Client library usually picks this up
    print(location)  # If you're setting via env

    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set.")
    if not location:
        raise ValueError("GOOGLE_CLOUD_LOCATION environment variable not set.")

    print(f"Initializing Vertex AI for Project: {project_id}, Location: {location}")

    try:
        # The vertexai.init() function uses GOOGLE_APPLICATION_CREDENTIALS automatically
        vertexai.init(project=project_id, location=location)
        print("Vertex AI initialized successfully.")
    except Exception as e:
        print(f"Error initializing Vertex AI: {e}")
        raise # Re-raise the exception to stop execution if init fails

# Example of how to use this in other files:
# from google_ai_utils import initialize_vertex_ai
# initialize_vertex_ai() # Call this once at the start of your application