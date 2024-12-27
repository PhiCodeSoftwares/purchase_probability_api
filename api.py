from flask import Flask, request, jsonify
import numpy as np
from model.hmm import *
from flask_cors import CORS

# Flask API
app = Flask(__name__)

CORS(app)

model = HMMModel(run=True)

@app.route("/api/options", methods=["GET"])
def get_options():
    try:
        options = {
            "options": model.get_observations()
        }

        return jsonify(options)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/purchase_accuracy", methods=["POST"])
def purchase_accuracy():
    try:
        # Get the observation sequence from the request
        data = request.json
        observation_sequence = data.get("observation_sequence")

        if not observation_sequence:
            return jsonify({"error": "Missing observation_sequence"}), 400

        # Convert the observation sequence to a numpy array
        observation_array = np.array(observation_sequence).reshape(-1, 1)

        # Predict probabilities
        prob_purchase, prob_no_purchase = model.predict_probabilities(observation_array)

        return jsonify({
            "probability_purchase": prob_purchase,
            "probability_no_purchase": prob_no_purchase
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(
        debug=False, 
        port=8000
    )