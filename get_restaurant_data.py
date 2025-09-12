from google.maps import places_v1
from google.type import latlng_pb2
import pandas as pd

API_KEY = "AIzaSyCvyaP2iLvENkLBAIqy_2hhyBVRJSrxVQk"
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

# Fixed function to combine categories with average ratings
def temp_category_combination(user1, user2):

    all_categories = {}
    for row in user1.itertuples():

        categories = row.Categories.split(", ")
        rating = row.Rating
        for category in categories:
            if category not in all_categories:
                all_categories[category] = rating
            else:
                all_categories[category] = (all_categories[category] + rating) / 2

    combined = dict(sorted(all_categories.items(), key=lambda item: item[1], reverse=True))
    return list(combined.keys())[:5]


async def output_recommendation(categories, location, price_level):
    if price_level == 1:
        price = places_v1.types.PriceLevel.PRICE_LEVEL_INEXPENSIVE
    elif price_level == 2:
        price = places_v1.types.PriceLevel.PRICE_LEVEL_MODERATE
    elif price_level == 3:
        price = places_v1.types.PriceLevel.PRICE_LEVEL_EXPENSIVE
    elif price_level == 4:
        price = places_v1.types.PriceLevel.PRICE_LEVEL_VERY_EXPENSIVE



    # Coordinates and radius for Columbus
    lat = 39.9625
    lng = -83.0032
    radius_meters = 50000.0 # ~30 mile radius


    # Create the LatLng object for the center
    center_point = latlng_pb2.LatLng(latitude=lat, longitude=lng)
    # Create the Circle object
    circle_area = places_v1.types.Circle(
        center=center_point,
        radius=radius_meters
    )
    # Create the location bias circle
    location_bias = places_v1.SearchTextRequest.LocationBias(
        circle=circle_area
    )
    # Define the search query and other parameters
    search_query =  str(categories)
    min_place_rating = 4.0
    client = places_v1.PlacesAsyncClient(
        client_options={"api_key": API_KEY}
    )
    # Build the request
    request = places_v1.SearchTextRequest(
        text_query=search_query,
        location_bias=location_bias,
        min_rating=min_place_rating,
        open_now=False,
        price_levels=[
            price
        ]
    )
    # Set the field mask
    fieldMask = "places.formattedAddress,places.displayName,places.id,places.types"
    # Make the request
    response = await client.search_text(request=request, metadata=[("x-goog-fieldmask",fieldMask)])
    
    list_of_ids = []
    list_of_places = []
    list_of_types = []

    for place in response.places:
        list_of_ids.append(place.id)
        list_of_places.append(place.display_name.text)
        list_of_types.append(place.types)

    rec_df = pd.DataFrame(
        {
            "ID": list_of_ids,
            "Name": list_of_places,
            "Types": list_of_types
        }
    )
    return rec_df