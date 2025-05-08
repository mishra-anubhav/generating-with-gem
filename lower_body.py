from serpapi.google_search import GoogleSearch
import os

def search_lower_body(item, gender, keyword="", num_results=10):
    query = f"{keyword} {gender} {item}".strip()
    params = {
        "api_key": os.getenv("SERPAPI_KEY"),
        "engine": "google_shopping",
        "q": query,
        "google_domain": "google.com",
        "num": num_results
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("shopping_results", [])
