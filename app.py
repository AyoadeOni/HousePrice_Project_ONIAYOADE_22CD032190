from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Get input values from form
            overall_qual = float(request.form["OverallQual"])
            gr_liv_area = float(request.form["GrLivArea"])
            total_bsmt_sf = float(request.form["TotalBsmtSF"])
            garage_cars = float(request.form["GarageCars"])
            full_bath = float(request.form["FullBath"])
            year_built = float(request.form["YearBuilt"])

            # Arrange input into array
            input_data = np.array([[ 
                overall_qual,
                gr_liv_area,
                total_bsmt_sf,
                garage_cars,
                full_bath,
                year_built
            ]])

            # Scale input
            input_scaled = scaler.transform(input_data)

            # Predict
            prediction = model.predict(input_scaled)[0]

        except Exception as e:
            prediction = "Error: Invalid input"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
