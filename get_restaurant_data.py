def get_restaurant_data(resName):
    client = places_v1.PlacesClient(
        client_options={"api_key": API_KEY}
    )
    
    # Coordinates of Columbus
    lat = 39.9625
    lng = -83.0032
    radius_meters = 50000.0  # ~30 mile radius

    # Create location bias
    center_point = latlng_pb2.LatLng(latitude=lat, longitude=lng)
    circle_area = places_v1.types.Circle(center=center_point, radius=radius_meters)
    location_bias = places_v1.SearchTextRequest.LocationBias(circle=circle_area)

    # Search for restaurant
    search_request = places_v1.SearchTextRequest(
        text_query=resName,
        location_bias=location_bias
    )
    field_mask = (
        "places.displayName,"
        "places.types,"
        "places.editorialSummary.text,"
        "places.reviewSummary.text"
    )
    search_response = client.search_text(
        request=search_request,
        metadata=[("x-goog-fieldmask", field_mask)],
    )

    if not search_response.places:
        return pd.DataFrame(columns=["Categories", "Reviews"])

    place = search_response.places[0]  # First match

    # Extract fields safely
    name = place.display_name.text if place.display_name else resName
    categories = ", ".join(place.types) if place.types else ""
    editorial = place.editorial_summary.text if place.editorial_summary else ""
    reviews = place.review_summary.text.text if place.review_summary else ""

    combined_reviews = (editorial + " " + reviews).strip()

    # Build DataFrame
    df = pd.DataFrame(
        {
            "Categories": [categories],
            "Reviews": [combined_reviews],
        },
        index=[name],
    )
    return df
