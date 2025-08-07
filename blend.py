import requests
from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

YELP_API_KEY = "mWUBF5oFXaHRmYO85IB4OroRnCZY_FZRWSMpNaI3Xxv3pSOIkgbed1Lat3-sYtBG2vBynsJb7Efk6U75khaZhs6v9gFs2ZSDxAntZ-DU-AC7xSBs271eXqDj-HiKaHYx"
HEADERS = {"Authorization": f"Bearer {YELP_API_KEY}"}

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
    user_location = "Columbus, OH"
    user_cuisines = ["Japanese", "Mexican", "Korean", "Vietnamese"]
    search_term = ",".join(user_cuisines)

    user1 = {
        "cuisines": ["Japanese", "Korean"],
        "dishes": ["sushi", "ramen"],
        "restaurants": ["Akai Hana"],
    }

    user2 = {
        "cuisines": ["Mexican", "Vietnamese"],
        "dishes": ["burrito", "noodles"],
        "restaurants": ["Buckeye Pho"]
    }

    restaurants = search_restaurants(user_location, search_term)
    high_quality = filter_restaurants(restaurants, min_rating=4.0)

    #for r in high_quality:
        #print(f"{r['name']} - {r['rating']} stars - {r['location']['address1']}")

    user1_embedding = get_embedding(user1)
    user2_embedding = get_embedding(user2)

    blended_embedding = (user1_embedding + user2_embedding) / 2
    #print(blended_embedding)

    best_score = -1
    best_restaurant = None

    for r in high_quality:
        restaurant_embedding = get_resteraunt_embedding(r)
        similarity = cosine_similarity(restaurant_embedding, blended_embedding)[0][0]  # Get scalar from 1x1 array
        if similarity > best_score:
            best_score = similarity
            best_restaurant = r

    if best_restaurant:
        print(f"🎯 Best Match: {best_restaurant['name']} - {best_restaurant['rating']} stars")
        print(f"Similarity Score: {best_score}")
    else:
        print("No matching restaurant found.")
    

    

# next step: use top 3 beli ratings from each user, weight the embedding for each user so that the beli rating is what its multiplied by