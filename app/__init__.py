from flask import Flask, render_template, request, jsonify, redirect, url_for
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configurations
app.config["APP_NAME"] = os.getenv("APP_NAME", "AI Chat")
app.config["MODEL"] = os.getenv("MODEL", "gpt-3.5-turbo")
app.config["ADMIN_USER"] = os.getenv("ADMIN_USER", "admin")
app.config["ADMIN_PASS"] = os.getenv("ADMIN_PASS", "password123")

@app.route("/")
def home():
    return render_template("index.html", app_name=app.config["APP_NAME"])

@app.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "POST":
        if (request.form.get("username") == app.config["ADMIN_USER"] and 
            request.form.get("password") == app.config["ADMIN_PASS"]):
            app.config["APP_NAME"] = request.form.get("app_name", "AI Chat")
            app.config["MODEL"] = request.form.get("model", "gpt-3.5-turbo")
            return redirect(url_for("home"))
        return "Invalid credentials", 401
    return render_template("admin.html")

@app.route("/chat", methods=["POST"])
def chat():
    if not request.json or "messages" not in request.json:
        return jsonify({"error": "Invalid request format"}), 400
    
    try:
        response = client.chat.completions.create(
            model=app.config["MODEL"],
            messages=request.json["messages"],
            temperature=0.7,
            max_tokens=500
        )
        
        if not response.choices:
            return jsonify({"error": "No response from AI"}), 500
            
        return jsonify({
            "content": response.choices[0].message.content,
            "model": app.config["MODEL"]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/check_api")
def check_api():
    try:
        # New way to check models in v1.0+
        models = client.models.list()
        return jsonify({
            "status": "success",
            "model_count": len(models.data)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)