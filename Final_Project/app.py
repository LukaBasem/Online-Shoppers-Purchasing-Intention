"""
Online Shoppers Purchase Intention - Gradio App
NTI Machine Learning for Data Analysis | Final Project

How it works:
1. Load the saved pipeline (preprocessing + Random Forest) from models/best_model.joblib
2. The user enters the details of a browsing session
3. We build one row of data, add the 3 engineered features, and let the pipeline predict
"""

import os

import gradio as gr
import joblib
import pandas as pd

# ------------------------------------------------------------------ load the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "models", "best_model.joblib"))

# ------------------------------------------------------------------ choices for dropdowns
MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]
SPECIAL_DAYS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
OPERATING_SYSTEMS = list(range(1, 9))
BROWSERS = list(range(1, 14))
REGIONS = list(range(1, 10))
TRAFFIC_TYPES = list(range(1, 21))


def to_number(value):
    """An empty box returns None, so we treat it as 0."""
    return 0.0 if value is None else float(value)


# ------------------------------------------------------------------ prediction function
def predict(administrative, administrative_duration, informational, informational_duration,
            product_related, product_related_duration, bounce_rates, exit_rates, page_values,
            special_day, month, operating_systems, browser, region, traffic_type,
            visitor_type, weekend):

    administrative = to_number(administrative)
    administrative_duration = to_number(administrative_duration)
    informational = to_number(informational)
    informational_duration = to_number(informational_duration)
    product_related = to_number(product_related)
    product_related_duration = to_number(product_related_duration)

    # same 3 engineered features that were created in the notebook (section 8)
    total_pages = administrative + informational + product_related
    total_duration = administrative_duration + informational_duration + product_related_duration
    avg_time_per_page = total_duration / total_pages if total_pages > 0 else 0

    row = pd.DataFrame([{
        "Administrative": administrative,
        "Administrative_Duration": administrative_duration,
        "Informational": informational,
        "Informational_Duration": informational_duration,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_related_duration,
        "BounceRates": to_number(bounce_rates),
        "ExitRates": to_number(exit_rates),
        "PageValues": to_number(page_values),
        "SpecialDay": float(special_day),
        "Month": month,
        "OperatingSystems": int(operating_systems),
        "Browser": int(browser),
        "Region": int(region),
        "TrafficType": int(traffic_type),
        "VisitorType": visitor_type,
        "Weekend": int(weekend),
        "TotalPages": total_pages,
        "TotalDuration": total_duration,
        "AvgTimePerPage": avg_time_per_page,
    }])

    # the pipeline does the scaling + encoding by itself, then the model predicts
    probability = float(model.predict_proba(row)[0][1])

    return {"Will BUY 🛒": probability, "Will NOT buy": 1 - probability}


# ------------------------------------------------------------------ user interface
with gr.Blocks(title="Online Shoppers Purchase Prediction") as demo:
    gr.Markdown(
        """
        # 🛒 Online Shoppers Purchase Prediction
        Enter the details of a browsing session and the model (Random Forest) will estimate
        **the probability that the visitor will buy something**.

        *NTI – Machine Learning for Data Analysis | Final Project*
        """
    )

    gr.Markdown("### 📄 Pages visited and time spent")
    with gr.Row():
        administrative = gr.Number(label="Administrative pages", value=0, minimum=0, precision=0)
        administrative_duration = gr.Number(label="Administrative duration (sec)", value=0, minimum=0)
        informational = gr.Number(label="Informational pages", value=0, minimum=0, precision=0)
    with gr.Row():
        informational_duration = gr.Number(label="Informational duration (sec)", value=0, minimum=0)
        product_related = gr.Number(label="Product pages", value=10, minimum=0, precision=0)
        product_related_duration = gr.Number(label="Product pages duration (sec)", value=300, minimum=0)

    gr.Markdown("### 📊 Behaviour metrics")
    with gr.Row():
        bounce_rates = gr.Number(label="Bounce rate (0 – 0.2)", value=0.02, minimum=0, maximum=0.2)
        exit_rates = gr.Number(label="Exit rate (0 – 0.2)", value=0.04, minimum=0, maximum=0.2)
        page_values = gr.Number(label="Page value", value=0, minimum=0)
        special_day = gr.Dropdown(label="Closeness to a special day (0 = far, 1 = very close)",
                                  choices=SPECIAL_DAYS, value=0.0)

    gr.Markdown("### 🌍 Visit context")
    with gr.Row():
        month = gr.Dropdown(label="Month", choices=MONTHS, value="Nov")
        visitor_type = gr.Dropdown(label="Visitor type", choices=VISITOR_TYPES, value="Returning_Visitor")
        weekend = gr.Checkbox(label="Weekend visit?", value=False)
    with gr.Row():
        operating_systems = gr.Dropdown(label="Operating system (code)", choices=OPERATING_SYSTEMS, value=2)
        browser = gr.Dropdown(label="Browser (code)", choices=BROWSERS, value=2)
        region = gr.Dropdown(label="Region (code)", choices=REGIONS, value=1)
        traffic_type = gr.Dropdown(label="Traffic type (code)", choices=TRAFFIC_TYPES, value=2)

    inputs = [administrative, administrative_duration, informational, informational_duration,
              product_related, product_related_duration, bounce_rates, exit_rates, page_values,
              special_day, month, operating_systems, browser, region, traffic_type,
              visitor_type, weekend]

    predict_button = gr.Button("🔮 Predict", variant="primary")
    output = gr.Label(label="Prediction", num_top_classes=2)

    predict_button.click(fn=predict, inputs=inputs, outputs=output)

    gr.Markdown("### Try a real example from the dataset")
    gr.Examples(
        examples=[
            # a session that ended with a purchase
            [0, 0, 0, 0, 13, 710.07, 0.0, 0.008, 72.52, 0.0, "Nov", 1, 1, 2, 2, "Returning_Visitor", False],
            # a session that ended without a purchase
            [0, 0, 0, 0, 19, 154.22, 0.016, 0.025, 0.0, 0.0, "Feb", 2, 2, 1, 3, "Returning_Visitor", False],
        ],
        inputs=inputs,
        outputs=output,
        fn=predict,
        label="Click a row, then press Predict",
    )

if __name__ == "__main__":
    demo.launch()
