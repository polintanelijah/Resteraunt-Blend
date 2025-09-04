def get_restaurant_embedding(resDataString):
    client = genai.Client(api_key=GEMINI_API_KEY)
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=resDataString,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )
    return np.array(result.embeddings[0].values)
