import requests
from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

YELP_API_KEY = "mWUBF5oFXaHRmYO85IB4OroRnCZY_FZRWSMpNaI3Xxv3pSOIkgbed1Lat3-sYtBG2vBynsJb7Efk6U75khaZhs6v9gFs2ZSDxAntZ-DU-AC7xSBs271eXqDj-HiKaHYx"
HEADERS = {"Authorization": f"Bearer {YELP_API_KEY}"}

def get_restaurant_data(name, location):
    url = "https://api.yelp.com/v3/businesses/search"
    params = {
        "term": name,
        "location": location,
        "limit": 1
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        raise Exception(f"Yelp API error: {response.status_code} {response.text}")
    businesses = response.json().get("businesses")
    return businesses[0] if businesses else None


def search_restaurants(location, term, radius_meters=32186, limit=50):
    url = "https://api.yelp.com/v3/businesses/search"
    params = {
        "location": location,        # or use lat/lon
        "term": term,                # e.g., "sushi, tacos"
        "radius": radius_meters,     # 20 miles = 32186 meters
        "limit": limit,
        "sort_by": "best_match"
    }

    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        raise Exception(f"Yelp API error: {response.status_code} {response.text}")

    data = response.json()
    return data["businesses"]

def get_categories_from_top_rated(user_ratings):
    all_categories = set()
    for restaurant_name in user_ratings.keys():
        data = get_restaurant_data(restaurant_name)
        if data and "categories" in data:
            categories = [c["title"] for c in data["categories"]]
            all_categories.update(categories)
    return list(all_categories)


def filter_restaurants(restaurants, min_rating=4.0):
    return [
        r for r in restaurants
        if r.get("rating", 0) >= min_rating
    ]


def get_embedding(user):
    client = genai.Client(api_key="AIzaSyCqgF4x2PCaWZM-SMP2oopL1mE8AxYxBgI")
    text = ", ".join(user['cuisines'] + user['dishes'] + user['restaurants'])


    result = [
        np.array(e.values) for e in client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")).embeddings
    ]

    # Calculate cosine similarity. Higher scores = greater semantic similarity.

    embeddings_matrix = np.array(result)
    #similarity_matrix = cosine_similarity(embeddings_matrix)

    return embeddings_matrix

def get_resteraunt_embedding(resteraunt):
    client = genai.Client(api_key="AIzaSyCqgF4x2PCaWZM-SMP2oopL1mE8AxYxBgI")
    categories = ", ".join([c['title'] for c in resteraunt['categories']])


    result = [
        np.array(e.values) for e in client.models.embed_content(
            model="gemini-embedding-001",
            contents=categories,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")).embeddings
    ]

    # Calculate cosine similarity. Higher scores = greater semantic similarity.

    embeddings_matrix = np.array(result)
    #similarity_matrix = cosine_similarity(embeddings_matrix)

    return embeddings_matrix







if __name__ == "__main__":
    # User embeddings and calculations
    user1 = {
        "Sushi Ten": 10.0,
        "Akai Hana": 9.7,
        "CM Chicken": 9.3
    }

    user2 = {
        "Song Lan": 10.0,
        "Sushi Ten": 9.6,
        "Valentinas": 9.1
    }

    # Step 1: Collect weighted embeddings
    weighted_sum = None
    total_weight = 0

    for r, rating in user1.items():
        embedding = get_resteraunt_embedding(r)  
        
        if weighted_sum is None:
            weighted_sum = embedding * rating
        else:
            weighted_sum += embedding * rating

        total_weight += rating

    # Step 2: Compute weighted average
    weighted_user1_embedding = weighted_sum / total_weight


        # Step 1: Collect weighted embeddings
    weighted_sum = None
    total_weight = 0

    for r, rating in user2.items():
        embedding = get_resteraunt_embedding(r)  
        
        if weighted_sum is None:
            weighted_sum = embedding * rating
        else:
            weighted_sum += embedding * rating

        total_weight += rating

    # Step 2: Compute weighted average
    weighted_user2_embedding = weighted_sum / total_weight

    blended_embedding = (weighted_user1_embedding + weighted_user2_embedding) / 2

    # Resteraunt embeddings
    user_location = "Columbus, OH"
    categories1 = get_categories_from_top_rated(user1)
    categories2 = get_categories_from_top_rated(user2)
    combined_categories = list(set(categories1 + categories2))
    search_term = ", ".join(combined_categories)


    restaurants = search_restaurants(user_location, search_term)
    high_quality = filter_restaurants(restaurants, min_rating=4.0)

   # Collect all similarity scores
    scored_restaurants = []

    for r in high_quality:
        restaurant_embedding = get_resteraunt_embedding(r)
        similarity = cosine_similarity(restaurant_embedding, blended_embedding)[0][0]
        scored_restaurants.append((similarity, r))

    # Sort by similarity (highest first) and get top 3
    top_matches = sorted(scored_restaurants, key=lambda x: x[0], reverse=True)[:3]

    # Output
    if top_matches:
        print("🎯 Top 3 Matches:")
        for i, (score, r) in enumerate(top_matches, 1):
            print(f"{i}. {r['name']} - {r['rating']} stars | Similarity: {score:.4f}")
    else:
        print("No matching restaurants found.")

    

    

# next step: use top 3 beli ratings from each user, weight the embedding for each user so that the beli rating is what its multiplied by