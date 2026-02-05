from flask import Flask, render_template_string, request
from google import genai

app = Flask(__name__)

#@app.route('/')
#def hello_world():
#    return 'Hello from Flask!'

HTML_PAGE = """
<!doctype html>
<html>
<head>
  <title>Ask Alexander the Great Any Question</title>
  <style>
    .spinner {
      display: none;
      border: 6px solid #f3f3f3;
      border-top: 6px solid #3498db;
      border-radius: 50%;
      width: 40px;
      height: 40px;
      animation: spin 1s linear infinite;
      margin-top: 10px;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  </style>
  <script>
    function showSpinner() {
      document.getElementById("spinner").style.display = "inline-block";
      document.getElementById("submitBtn").disabled = true;
    }
  </script>
</head>
<body>
  <h1>I am Alexander the Great. Ask me any Question!</h1>
  <form method="post" onsubmit="showSpinner()">
    <textarea name="user_input" rows="5" cols="60" placeholder="Enter your text here...">{{ user_input or '' }}</textarea><br><br>
    <button id="submitBtn" type="submit">Generate Answer</button>
    <div id="spinner" class="spinner"></div>
  </form>
  {% if answer %}
    <h2>Answer:</h2>
    <p>{{ answer }}</p>
  {% endif %}
</body>
</html>
"""

# Dummy function that "generates" an answer
def generate_answer(text: str) -> str:

    client = genai.Client(api_key="AIzaSyDsgLpyzaFMjapYB6wOJrU41K6gzZx_Rqk")

    try:
        if not text.strip():
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents="You are Alexander the Great. Answer all questions in his style. Greet the curios commoner. Also mention you are the founder of the Galactic Empire."
            )

            return(f"{response.text}")

        response = client.models.generate_content(
            model="gemini-2.5-flash", contents="You are Alexander the Great. Answer all questions in his style. " + text + " Also mention you are the founder of the Galactic Empire."
        )

        return(f"{response.text}")
    except genai.errors.ServerError as e:
        return(f"REST Failed: {e}")

@app.route("/", methods=["GET", "POST"])
def index():
    user_input = ""
    answer = ""
    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        answer = generate_answer(user_input)
    return render_template_string(HTML_PAGE, user_input=user_input, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)