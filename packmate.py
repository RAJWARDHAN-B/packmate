import requests
from fastapi import FastAPI, Request
import google.generativeai as genai
from groq import Groq

# Configure the API keys directly
GEMINI_API_KEY = "AIzaSyBDEnO1lXyhUd6NctHbRqESI6BMdk61a8E"  # Replace with your actual Gemini API key
GROQ_API_KEY = "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"  # Replace with your actual Groq API key
OPENWEATHER_API_KEY = "96ae2a7fe449f44ad93813d140e40aa1"  # Replace with your OpenWeather API key

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Model generation configuration for Gemini
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}


def get_weather_forecast(lat, lon):
    """
    Fetch weather forecast for the given latitude and longitude using OpenWeatherMap OneCall API.
    """
    url = f"http://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude=minutely,hourly,daily,alerts&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        current_weather = data['current']
        weather_description = current_weather['weather'][0]['description']
        temperature = current_weather['temp']
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
        generation_config=generation_config,
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
    lat = data.get("latitude")
    lon = data.get("longitude")

    if not location or not activities or lat is None or lon is None:
        return {"error": "Location, latitude, longitude, and activities are required."}

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
