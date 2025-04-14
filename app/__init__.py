from flask import Flask, render_template, request, jsonify, redirect, url_for
import openai
import os
import json

app = Flask(__name__)

# Load config (this is in Render's environment variables)
app.config["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
app.config["APP_NAME"] = os.getenv("APP_NAME", "AI Chat")
app.config["MODEL"] = os.getenv("MODEL", "gpt-3.5-turbo")

# Admin credentials (change these in Render env vars)
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "password123")

@app.route("/")
def home():
    return render_template("index.html", app_name=app.config["APP_NAME"])

@app.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "POST":
        # Basic auth (not for production, use Flask-Login in real apps)
        if (
            request.form.get("username") == ADMIN_USER
            and request.form.get("password") == ADMIN_PASS
        ):
            # Update settings
            app.config["APP_NAME"] = request.form.get("app_name", "AI Chat")
            app.config["MODEL"] = request.form.get("model", "gpt-3.5-turbo")
            return redirect(url_for("home"))
        else:
            return "Invalid credentials", 401
    return render_template("admin.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    try:
        response = openai.ChatCompletion.create(
            model=app.config["MODEL"],
            messages=data["messages"],
        )
        # Ensure we're returning the content properly
        return jsonify({
            "content": response.choices[0].message['content']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
if __name__ == "__main__":
    app.run(debug=True)
