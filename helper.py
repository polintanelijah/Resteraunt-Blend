import requests
from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

YELP_API_KEY = ""
HEADERS = {"Authorization": f"Bearer {YELP_API_KEY}"}
YELP_BASE_URL = "https://api.yelp.com/v3/businesses/search"

GEMINI_API_KEY = ""

def get_restaurant_data(name, location):
    url = "https://api.yelp.com/v3/businesses/search"
    params = {"term": name, "location": location, "limit": 1}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        raise Exception(f"Yelp API error: {response.status_code} {response.text}")
    businesses = response.json().get("businesses")
    return businesses[0] if businesses else None

def search_restaurants_with_price(location, price, term, radius_meters=40000, limit=50):
    url = "https://api.yelp.com/v3/businesses/search"
    params = {
        "location": location,
        "term": term,
        "radius": radius_meters,
        "limit": limit,
        "sort_by": "best_match",
        "price": price
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        raise Exception(f"Yelp API error: {response.status_code} {response.text}")
    return response.json().get("businesses", [])

def search_restaurants(location, term, radius_meters=40000, limit=10):
    url = "https://api.yelp.com/v3/businesses/search"
    params = {
        "location": location,
        "term": term,
        "radius": radius_meters,
        "limit": limit,
        "sort_by": "best_match",
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        raise Exception(f"Yelp API error: {response.status_code} {response.text}")
    return response.json().get("businesses", [])


def get_categories_from_top_rated(user_ratings, location):
    all_categories = set()
    for restaurant_name in user_ratings:
        data = get_restaurant_data(restaurant_name, location)
        if data and "categories" in data:
            all_categories.update(c["title"] for c in data["categories"])
    return list(all_categories)

def filter_restaurants(restaurants, min_rating=3.5):
    return [r for r in restaurants if r.get("rating", 0) >= min_rating]

def get_restaurant_embedding(restaurant):
    client = genai.Client(api_key=GEMINI_API_KEY)
    categories = ", ".join([c["title"] for c in restaurant.get("categories", [])])
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=categories,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )
    return np.array(result.embeddings[0].values)

def weighted_embedding(user_ratings, location):
    client = genai.Client(api_key=GEMINI_API_KEY)
    weighted_sum = None
    total_weight = 0
    for name, rating in user_ratings.items():
        data = search_restaurants(location, name)
        if not data:
            print(f"❌ Could not find: {name}")
            continue
        restaurant = data[0]
        embedding = get_restaurant_embedding(restaurant)
        if weighted_sum is None:
            weighted_sum = embedding * rating
        else:
            weighted_sum += embedding * rating
        total_weight += rating
    return weighted_sum / total_weight if total_weight else None

def find_restaurant_exact(name, location):
    """Search Yelp for an exact restaurant name match in a given location."""
    params = {
        "term": name,
        "location": location,
        "limit": 10  # get more results so we can find exact match
    }
    response = requests.get(YELP_BASE_URL, headers=HEADERS, params=params)
    data = response.json()

    if "businesses" not in data:
        return None

    # Exact match ignoring case
    for biz in data["businesses"]:
        if biz["name"].strip().lower() == name.strip().lower():
            return biz

    return None  # no exact match found


def get_categories_from_restaurants(restaurant_names, location):
    """Return list of category titles from the found restaurants."""
    all_categories = []
    for name in restaurant_names:
        r = search_restaurants(location, name)
        restaurant = r[0]
        if restaurant:
            categories = [c["title"] for c in restaurant.get("categories", [])]
            all_categories.extend(categories)
        else:
            print(f"⚠️ No exact match found for '{name}'")
    return list(set(all_categories))


def get_recommendations(categories, location, price=None):
    """Search Yelp for restaurants matching given categories."""
    if not categories:
        return []

    params = {
        "categories": ",".join([c.lower().replace(" ", "") for c in categories]),
        "location": location,
        "limit": 10
    }
    if price:
        params["price"] = price  # Yelp price levels: 1, 2, 3, 4

    response = requests.get(YELP_BASE_URL, headers=HEADERS, params=params)
    data = response.json()
    return data.get("businesses", [])