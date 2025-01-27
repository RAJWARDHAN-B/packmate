import requests
from fastapi import FastAPI, Request
import google.generativeai as genai
from groq import Groq
import time
from datetime import datetime, timedelta
from meteostat import Daily, Point

# Configure API keys
GEMINI_API_KEY = "AIzaSyBDEnO1lXyhUd6NctHbRqESI6BMdk61a8E"
GROQ_API_KEY = "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"
OPENWEATHER_API_KEY = "f65ed6d0474abe2144b0d4766558ea99"
OPENCAGE_API_KEY = "993e21d6cca746a2bbebd2f6e02a8316"

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Cache for storing location coordinates to reduce API calls
location_cache = {}

# Rate limit: Avoid making too many requests in a short time
RATE_LIMIT_SECONDS = 1  # Add a delay of 1 second between requests


def get_lat_lon_from_nominatim(location):
    """Fetch latitude and longitude using Nominatim Geocoding API."""
    if location in location_cache:
        return location_cache[location]

    from urllib.parse import quote
    location_encoded = quote(location)
    url = f"https://nominatim.openstreetmap.org/search?q={location_encoded}&format=json&addressdetails=1"
    headers = {"User-Agent": "YourAppName/1.0 (your_email@example.com)"}
    time.sleep(RATE_LIMIT_SECONDS)
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data:
            lat = data[0]["lat"]
            lon = data[0]["lon"]
            location_cache[location] = (lat, lon)
            return lat, lon
    return None, None


def get_weather_forecast(lat, lon, input_date):
    """
    Fetch weather data based on the date:
    - Within 7 days: Use OpenWeather API.
    - After 7 days: Use Meteostat for future weather data.
    """
    input_date = datetime.strptime(input_date, "%Y-%m-%d").date()
    today = datetime.now().date()

    # Case 1: Within 7 days (OpenWeather API)
    if today <= input_date <= today + timedelta(days=7):
        url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            delta_days = (input_date - today).days
            daily_weather = data["daily"][delta_days]
            temp = daily_weather["temp"]["day"]
            conditions = daily_weather["weather"][0]["description"]
            return f"Forecast for {input_date}: {conditions}, Temp: {temp}°C"
        else:
            return "Error fetching weather data from OpenWeather."

    # Case 2: Future dates beyond 7 days (Meteostat API)
    elif input_date > today + timedelta(days=7):
        location = Point(lat, lon)
        data = Daily(location, today + timedelta(days=7), input_date)
        data = data.fetch()
        if not data.empty:
            avg_temp = data["tavg"].iloc[0]
            return f"Forecast for {input_date}: Avg Temp: {avg_temp}°C"
        else:
            return f"No forecast data available for {input_date} from Meteostat."

    # Case 3: Handle other cases
    else:
        return "Weather data for this date is unavailable."


def get_packing_suggestions(location, activities):
    """Use the Google Generative AI library to generate packing suggestions."""
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config={"temperature": 1, "top_p": 0.95, "top_k": 40, "max_output_tokens": 8192},
    )
    chat_session = model.start_chat(history=[])
    prompt = f"Suggest a packing list for {activities} in {location}."
    try:
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        print("Error fetching packing suggestions from Gemini API:", e)
        return None


def fallback_packing_suggestions(location, activities):
    """Use Groq API as a fallback for packing suggestions."""
    client = Groq(api_key=GROQ_API_KEY)
    query = f"Suggest a packing list for {activities} in {location}."
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": query}],
            temperature=1,
            max_completion_tokens=200,
        )
        response_content = completion.choices[0].message.get("content", "")
        return response_content if response_content else "No suggestions available."
    except Exception as e:
        print("Failed to fetch packing suggestions from Groq API:", e)
        return "No suggestions available."


@app.post("/generate_packing_list")
async def generate_packing_list(request: Request):
    """Generate a packing list based on location, date, weather, and activities."""
    data = await request.json()
    location = data.get("location")
    activities = data.get("activities", [])
    input_date = data.get("date", datetime.now().strftime("%Y-%m-%d"))

    if not location or not activities:
        return {"error": "Location and activities are required."}

    # Get latitude and longitude
    lat, lon = get_lat_lon_from_nominatim(location)
    if lat is None or lon is None:
        lat, lon = get_lat_lon_from_opencage(location)

    if lat is None or lon is None:
        return {"error": f"Unable to find coordinates for {location}."}

    # Fetch weather data
    weather_data = get_weather_forecast(lat, lon, input_date)

    # Generate packing suggestions
    packing_suggestions = get_packing_suggestions(location, activities)
    if packing_suggestions:
        return {
            "weather": weather_data,
            "packing_list": packing_suggestions.split("\n"),
        }

    # Fallback to Groq
    groq_suggestions = fallback_packing_suggestions(location, activities)
    return {
        "weather": weather_data,
        "packing_list": groq_suggestions.split("\n") if groq_suggestions else [],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
