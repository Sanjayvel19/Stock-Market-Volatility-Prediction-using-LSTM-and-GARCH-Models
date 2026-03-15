from flask import Flask, render_template, jsonify, request
import os
import joblib
import pandas as pd

app = Flask(__name__)

# ---------------- PATHS ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models")

lstm_results = None
garch_results = None
raw_rankings = []


# ---------------- LOAD MODELS ---------------- #

def load_models():
    global lstm_results, garch_results, raw_rankings

    try:

        lstm_file = os.path.join(MODEL_PATH, "lstm_results.pkl")
        garch_file = os.path.join(MODEL_PATH, "garch_results.pkl")

        if os.path.exists(lstm_file):
            lstm_results = joblib.load(lstm_file)

        if os.path.exists(garch_file):
            garch_results = joblib.load(garch_file)

        # Convert to dataframe if needed
        if isinstance(lstm_results, list):
            lstm_df = pd.DataFrame(lstm_results)
        else:
            lstm_df = lstm_results

        if isinstance(garch_results, list):
            garch_df = pd.DataFrame(garch_results)
        else:
            garch_df = garch_results

        if lstm_df is not None and garch_df is not None:

            combined = pd.merge(
                lstm_df,
                garch_df,
                on="Stock",
                how="inner"
            )

            combined["Combined Volatility"] = (
                combined["LSTM Volatility"] + combined["GARCH Volatility"]
            ) / 2

            combined = combined.sort_values(
                by="Combined Volatility",
                ascending=False
            )

            raw_rankings = combined.to_dict(orient="records")

            print("Models loaded successfully")

    except Exception as e:
        print("Error loading models:", e)
        raw_rankings = []


# Load models on startup
load_models()


# ---------------- ROUTES ---------------- #

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ---------------- API ---------------- #

@app.route("/api/dashboard_data")
def get_dashboard_data():

    global raw_rankings

    selected_stock = request.args.get("stock", "").strip()

    # If rankings empty
    if not raw_rankings:
        return jsonify({
            "error": "No stock ranking data available"
        })

    # If no stock provided → use first stock
    if selected_stock == "":
        selected_stock = raw_rankings[0]["Stock"]

    # Find selected stock
    target = next(
        (i for i in raw_rankings if i["Stock"] == selected_stock),
        raw_rankings[0]
    )

    # Top 10 rankings
    top10 = raw_rankings[:10]

    response = {
        "selected_stock": target,
        "top10": top10,
        "stocks": [i["Stock"] for i in raw_rankings]
    }

    return jsonify(response)


# ---------------- START SERVER ---------------- #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
