from flask import Flask, request, jsonify
from transformers import pipeline, BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration

app = Flask(__name__)

# Load model (cache it locally)
model = BlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")

def lucy_response(input_text):
    inputs = tokenizer([input_text], return_tensors="pt")
    reply_ids = model.generate(**inputs)
    return tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    response = lucy_response(user_input)
    return jsonify({"response": f"Lucy: {response}"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
