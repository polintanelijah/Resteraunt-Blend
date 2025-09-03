from google.maps import places_v1


API_KEY = ""

def get_restaurant_data(resName):
    client = places_v1.PlacesClient(
        client_options={"api_key": API_KEY}
    )

    search_request = places_v1.SearchTextRequest(
        text_query=resName,
    )
    field_mask = "places.formattedAddress,places.types,places.editorialSummary"
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
