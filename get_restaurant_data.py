from google.maps import places_v1
from google.type import latlng_pb2

API_KEY = ""

def get_restaurant_data(resName):
    client = places_v1.PlacesClient(
        client_options={"api_key": API_KEY}
    )
    
    # Coordinates of Columbus
    lat = 39.9625
    lng = -83.0032
    radius_meters = 50000.0     # ~30 mile radius

    # Create location bias
    center_point = latlng_pb2.LatLng(latitude=lat, longitude=lng)
    circle_area = places_v1.types.Circle(
        center=center_point,
        radius=radius_meters
    )
    location_bias = places_v1.SearchTextRequest.LocationBias(
        circle=circle_area
    )

    # Search for restaurant
    search_request = places_v1.SearchTextRequest(
        text_query=resName,
        location_bias=location_bias
    )
    field_mask = "places.formattedAddress,places.types,places.editorialSummary" # Data wanted
    search_response = client.search_text(
        request=search_request,
        metadata=[("x-goog-fieldmask", field_mask)],
    )

    if not search_response.places:
        return f"No results found for {resName}"

    place = search_response.places[0]  # First match
    return {
        "address": place.formatted_address,
        "types": place.types,
        "description": place.editorial_summary
    }
