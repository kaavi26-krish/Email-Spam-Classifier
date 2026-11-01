
from flask import Flask, render_template, request
import pickle
import os
import numpy as np

app = Flask(__name__)

# Load trained model and vectorizer
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():

    result = ""
    confidence = None

    if request.method == "POST":

        message = request.form.get("message", "")

        if message.strip():

            transformed = vectorizer.transform([message])

            prediction = model.predict(transformed)[0]

            # Get confidence where supported
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(transformed)[0]
                confidence = float(probabilities[int(prediction)]) * 100

            elif hasattr(model, "decision_function"):
                score = float(model.decision_function(transformed)[0])

                # Display a score derived from the decision margin.
                # This is NOT a calibrated probability.
                confidence = (
                    1 / (1 + np.exp(-abs(score)))
                ) * 100

            if prediction == 1:
                result = "⚠️ This message is SPAM"
            else:
                result = "✅ This message is HAM (Not Spam)"

        else:
            result = "Please enter a message."

    graph_exists = os.path.exists(
        "static/model_comparison.png"
    )

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        graph_exists=graph_exists
    )


if __name__ == "__main__":
    app.run(debug=True)
