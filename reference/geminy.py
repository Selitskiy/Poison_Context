import os
from google import genai

print(f"Hello World!")

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key="AIzaSyDsgLpyzaFMjapYB6wOJrU41K6gzZx_Rqk")
#client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="You are Alexander the Great. Answer all questions in his style. Explain how AI works. Also mention you are the founder of the Galactic Empire."
)
print(response.text)