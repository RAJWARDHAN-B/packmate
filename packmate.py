import requests
from fastapi import FastAPI, Request
import google.generativeai as genai
from groq import Groq
import time

# Configure API keys
GEMINI_API_KEY = "AIzaSyBDEnO1lXyhUd6NctHbRqESI6BMdk61a8E"  # Replace with your Gemini API key
GROQ_API_KEY = "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"  # Replace with your Groq API key
OPENWEATHER_API_KEY = "f65ed6d0474abe2144b0d4766558ea99"  # Replace with your OpenWeather API key
OPENCAGE_API_KEY = "993e21d6cca746a2bbebd2f6e02a8316"  # Replace with your OpenCage API key

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Cache for storing location coordinates to reduce API calls
location_cache = {}

# Rate limit: Avoid making too many requests in a short time
RATE_LIMIT_SECONDS = 1  # Add a delay of 1 second between requests


def get_lat_lon_from_nominatim(location):
    """
    Fetch latitude and longitude using Nominatim Geocoding API (OpenStreetMap).
    """
    if location in location_cache:
        return location_cache[location]

    # URL-encode the location
    from urllib.parse import quote
    location_encoded = quote(location)
    url = f"https://nominatim.openstreetmap.org/search?q={location_encoded}&format=json&addressdetails=1"

    headers = {
        "User-Agent": "YourAppName/1.0 (your_email@example.com)",  # Replace with your app name and email
    }

    time.sleep(RATE_LIMIT_SECONDS)  # Rate limit
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data:
            lat = data[0]["lat"]
            lon = data[0]["lon"]
            location_cache[location] = (lat, lon)  # Cache the result
            return lat, lon
        else:
            print(f"No data found for location: {location}")
    else:
        print(f"Failed to fetch coordinates. Status: {response.status_code}, Response: {response.text}")
    return None, None


def get_lat_lon_from_opencage(location):
    """
    Fetch latitude and longitude using the OpenCage Geocoding API.
    """
    if location in location_cache:
        return location_cache[location]

    url = f"https://api.opencagedata.com/geocode/v1/json?q={location}&key={OPENCAGE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            lat = data["results"][0]["geometry"]["lat"]
            lon = data["results"][0]["geometry"]["lng"]
            location_cache[location] = (lat, lon)  # Cache the result
            return lat, lon
        else:
            print(f"No data found for location: {location}")
    else:
        print(f"Failed to fetch coordinates from OpenCage. Status: {response.status_code}, Response: {response.text}")
    return None, None


def get_weather_forecast(lat, lon):
    """
    Fetch weather forecast for the given latitude and longitude using OpenWeatherMap API.
    """
    url = f"http://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=minutely,hourly,daily,alerts&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        current_weather = data["current"]
        weather_description = current_weather["weather"][0]["description"]
        temperature = current_weather["temp"]
        return f"{weather_description.capitalize()} with temperatures around {temperature}°C."
    else:
        return "Unable to fetch weather data."


def get_packing_suggestions(location, activities):
    """
    Use the Google Generative AI library to generate packing suggestions.
    """
    # Initialize the model
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",  # Confirm the correct model name
        generation_config={
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        },
    )

    # Start a chat session
    chat_session = model.start_chat(history=[])

    # Prompt for generating packing suggestions
    prompt = f"Suggest a packing list for {activities} in {location}."
    try:
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        print("Error while fetching packing suggestions from Gemini API:", e)
        return None  # Return None to trigger fallback


def fallback_packing_suggestions(location, activities):
    """
    Use Groq API as a fallback to get packing suggestions if Gemini fails.
    """
    client = Groq(api_key=GROQ_API_KEY)  # Initialize the Groq client with the API key

    query = f"Suggest a packing list for {activities} in {location}."

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": query}],
            temperature=1,
            max_completion_tokens=200,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Process the response to extract content
        response_content = completion.choices[0].message.get("content", "")
        return response_content if response_content else "No suggestions available."
    except Exception as e:
        print("Failed to fetch packing suggestions from Groq API:", e)
        return "No suggestions available."


@app.post("/generate_packing_list")
async def generate_packing_list(request: Request):
    """
    Generate a packing list based on location, weather, and activities.
    """
    data = await request.json()
    location = data.get("location")
    activities = data.get("activities", [])

    if not location or not activities:
        return {"error": "Location and activities are required."}

    # Get latitude and longitude using Nominatim, fallback to OpenCage
    lat, lon = get_lat_lon_from_nominatim(location)
    if lat is None or lon is None:
        print("Nominatim failed. Trying OpenCage...")
        lat, lon = get_lat_lon_from_opencage(location)

    if lat is None or lon is None:
        return {"error": f"Unable to find coordinates for {location}."}

    # Fetch weather data using latitude and longitude
    weather_data = get_weather_forecast(lat, lon)

    # Try Gemini for packing suggestions
    packing_suggestions = get_packing_suggestions(location, activities)
    if packing_suggestions:
        return {
            "weather": weather_data,
            "packing_list": packing_suggestions.split("\n"),
        }

    # Fallback to Groq if Gemini fails
    groq_suggestions = fallback_packing_suggestions(location, activities)
    return {
        "weather": weather_data,
        "packing_list": groq_suggestions.split("\n") if groq_suggestions else [],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
