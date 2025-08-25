import requests
from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from helper import *

YELP_API_KEY = ""
HEADERS = {"Authorization": f"Bearer {YELP_API_KEY}"}

GEMINI_API_KEY = ""

if __name__ == "__main__":
    user1_location = input("User 1 Location: ")
    user2_location = input("User 2 Location: ")

    # === Get inputs ===
    user1 = {}
    user2 = {}
    for i in range(3):
        r = input("User 1 - Enter restaurant: ")
        rating = float(input("User 1 - Enter rating: "))
        user1[r] = rating

    for i in range(3):
        r = input("User 2 - Enter restaurant: ")
        rating = float(input("User 2 - Enter rating: "))
        user2[r] = rating

    excluded = set(user1.keys()) | set(user2.keys())

    # === Embeddings ===
    emb1 = weighted_embedding(user1, user1_location)
    emb2 = weighted_embedding(user2, user2_location)

    if emb1 is None or emb2 is None:
        print("❌ Could not compute embeddings for both users.")
        exit()

    #blended_embedding = (emb1 + emb2) / 2

    # === Categories ===
    categories1 = get_categories_from_restaurants(user1, user1_location)
    categories2 = get_categories_from_restaurants(user2, user2_location)
    combined_categories = combine_categories_from_restaurants(categories1, categories2)

    print("🔍 Search categories:", combined_categories)
    price = input("Restaurant price levels (1, 2, 3, 4): ")

    # === Recommendations ===
    restaurants = get_recommendations(combined_categories, user1_location, price)
    print(f"🔍 Found {len(restaurants)} candidate restaurants.")

    # === Filter + Score ===
    scored = []
    for r in restaurants:
        if r["name"] in excluded or r.get("rating", 0) < 4.0:
            continue
        try:
            embedding = get_restaurant_embedding(r)
            sim1 = cosine_similarity([embedding], [emb1])[0][0]
            sim2 = cosine_similarity([embedding], [emb2])[0][0]
            min_sim = min(sim1, sim2)
            scored.append((min_sim, r))
        except Exception as e:
            print(f"⚠️ Skipping {r['name']}: {e}")

    top_matches = sorted(scored, key=lambda x: x[0], reverse=True)[:3]

    if top_matches:
        print("\n🎯 Top 3 Matches:")
        for i, (score, r) in enumerate(top_matches, 1):
            print(f"{i}. {r['name']} - {r['rating']}⭐ | Similarity: {score:.4f}")
    else:
        print("No matching restaurants found.")
