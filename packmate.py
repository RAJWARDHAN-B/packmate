import requests
from fastapi import FastAPI, Request
import google.generativeai as genai
from groq import Groq
import time
from meteostat import Daily
from datetime import datetime, timedelta

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


def get_weather_forecast(lat, lon, input_date):
    """
    Fetch weather forecast for the given latitude, longitude, and input date.
    Uses OpenWeather if within 7 days, otherwise uses Meteostat for future dates.
    """
    try:
        # Ensure lat and lon are floats
        lat = float(lat)
        lon = float(lon)

        # Convert input date to datetime object
        input_date_obj = datetime.strptime(input_date, "%Y-%m-%d")

        # Check if input date is within 7 days from today
        today = datetime.today()
        if input_date_obj <= today + timedelta(days=7):
            # Use OpenWeather API for weather data within 7 days
            weather_data = get_openweather_data(lat, lon)  # Implement this method for OpenWeather
        else:
            # Use Meteostat API for future dates beyond 7 days
            weather_data = get_meteostat_data(lat, lon, input_date_obj)

        return weather_data
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return "Unable to fetch weather data."


def get_openweather_data(lat, lon):
    """
    Fetch weather data from OpenWeather API for the given latitude and longitude.
    """
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        # Extract weather details
        weather_description = data["weather"][0]["description"]  # E.g., "clear sky"
        temperature = data["main"]["temp"]  # Current temperature in °C
        feels_like = data["main"]["feels_like"]  # Feels like temperature

        # Format the weather report
        return (
            f"Current weather is {weather_description.capitalize()} with a temperature "
            f"of {temperature}°C (feels like {feels_like}°C)."
        )
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return "Unable to fetch weather data."


from meteostat import Point, Daily
from datetime import datetime

def get_meteostat_data(lat, lon, input_date):
    """
    Fetch weather data using Meteostat for future dates beyond 7 days.
    """
    try:
        # Ensure input_date is a datetime object
        if isinstance(input_date, str):
            input_date = datetime.strptime(input_date, "%Y-%m-%d")

        # Create a Point object for the given latitude and longitude
        location = Point(lat, lon)

        # Set the time range for the input date (start and end are the same for a single day)
        daily = Daily(location, input_date, input_date)
        data = daily.fetch()

        # Extract relevant weather information
        if data.empty:
            return "No data available for the given date."

        # Example: Extract the average temperature
        avg_temp = data['tavg'].iloc[0]  # Get the average temperature for the day
        return f"Forecast for {input_date.strftime('%Y-%m-%d')}: Avg Temp: {avg_temp}°C"

    except Exception as e:
        print(f"Error fetching Meteostat data: {e}")
        return "Unable to fetch weather data from Meteostat."


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
    input_date = data.get("date")

    if not location or not activities or not input_date:
        return {"error": "Location, activities, and date are required."}

    # Get latitude and longitude using Nominatim, fallback to OpenCage
    lat, lon = get_lat_lon_from_nominatim(location)
    if lat is None or lon is None:
        print("Nominatim failed. Trying OpenCage...")
        lat, lon = get_lat_lon_from_opencage(location)

    if lat is None or lon is None:
        return {"error": f"Unable to find coordinates for {location}."}

    # Fetch weather data using latitude and longitude and the input date
    weather_data = get_weather_forecast(lat, lon, input_date)

    # Try Gemini for packing suggestions
    packing_suggestions = get_packing_suggestions(location, activities)
    if packing_suggestions:
        # Extract only the item names (without descriptions)
        packing_list = [item.strip() for item in packing_suggestions.split("\n") if item.strip()]
        return {
            "weather": weather_data,
            "packing_list": packing_list,
        }

    # Fallback to Groq if Gemini fails
    groq_suggestions = fallback_packing_suggestions(location, activities)
    packing_list = [item.strip() for item in groq_suggestions.split("\n") if item.strip()]
    
    return {
        "weather": weather_data,
        "packing_list": packing_list if packing_list else [],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
