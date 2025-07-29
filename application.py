from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict/", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        headline = request.form.get("headline")

        if headline:

            custom_data = CustomData(headline)
            input_df = custom_data.get_data_as_dataframe()

            predict_pipeline = PredictPipeline()
            prediction, confidence = predict_pipeline.predict(input_df)

            prediction = f"{prediction}"
            confidence = f"{round(confidence, 2)}%"
            
            return render_template("home.html", prediction = prediction, confidence = confidence)
    
    return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")