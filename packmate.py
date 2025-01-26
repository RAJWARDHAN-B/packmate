import requests
import os
from fastapi import FastAPI, Request

# Initialize FastAPI app
app = FastAPI()

def get_weather_forecast(location):
    """
    Fetch weather forecast for the given location using Gemini API.
    """
    GEMINI_API_KEY = "AIzaSyCClZrazhQxLcDBJRpHyhXuKCn8UCQZw4w"  # Fetch API key from environment variables
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key not set in environment variables.")

    GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [{
            "parts": [{"text": f"Get current weather data for {location}."}]
        }]
    }

    response = requests.post(GEMINI_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result.get("contents", [{}])[0].get("parts", [{}])[0].get("text", "No data available.")
    else:
        print("Failed to fetch weather data from Gemini API:", response.text)
        return None


def get_packing_suggestions(location, activities):
    """
    Use Gemini API to analyze location, weather, and activities for packing suggestions.
    """
    GEMINI_API_KEY = os.getenv("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")  # Fetch API key from environment variables
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key not set in environment variables.")

    GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "contents": [{
            "parts": [{"text": f"Suggest a packing list for {activities} in {location}."}]
        }]
    }

    response = requests.post(GEMINI_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result.get("contents", [{}])[0].get("parts", [{}])[0].get("text", "No suggestions available.")
    else:
        print("Failed to fetch packing suggestions from Gemini API:", response.text)
        return None


def fallback_packing_suggestions(location, activities):
    """
    Use Groq API as a fallback to get packing suggestions if Gemini fails.
    """
    from groq import Groq

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Fetch Groq API key from environment variables
    if not GROQ_API_KEY:
        raise ValueError("Groq API key not set in environment variables.")

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
        return None


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

    # Fetch weather data (optional, can be used for future extensions)
    weather_data = get_weather_forecast(location)
    if not weather_data:
        return {"error": "Could not retrieve weather information."}

    # Try Gemini for packing suggestions
    gemini_suggestions = get_packing_suggestions(location, activities)
    if gemini_suggestions:
        return {"packing_list": gemini_suggestions.split("\n")}

    # Fallback to Groq if Gemini fails
    groq_suggestions = fallback_packing_suggestions(location, activities)
    return {"packing_list": groq_suggestions.split("\n") if groq_suggestions else []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
