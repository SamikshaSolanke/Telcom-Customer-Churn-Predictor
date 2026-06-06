# # app.py code
# from fastapi import FastAPI
# from pydantic import BaseModel
# import gradio as gr
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from serving.inference import predict  

# app = FastAPI()
# @app.get("/")
# def root():
#     return {"status": "ok"}

# class CustomerData(BaseModel):
#     gender: str
#     Partner: str
#     Dependents: str
#     PhoneService: str
#     MultipleLines: str
#     InternetService: str
#     OnlineSecurity: str
#     OnlineBackup: str
#     DeviceProtection: str
#     TechSupport: str
#     StreamingTV: str
#     StreamingMovies: str
#     Contract: str
#     PaperlessBilling: str
#     PaymentMethod: str
#     tenure: int
#     MonthlyCharges: float
#     TotalCharges: float

# @app.post("/predict")
# def api_predict(data: CustomerData):
#     try:
#         out = predict(data.dict())
#         return {"prediction": out}
#     except Exception as e:
#         return {"error": str(e)}

# # --- Gradio UI wrappers the same predict() ---
# def gradio_interface(
#     gender, Partner, Dependents, PhoneService, MultipleLines,
#     InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
#     TechSupport, StreamingTV, StreamingMovies, Contract,
#     PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
# ):
#     payload = {
#         "gender": gender,
#         "Partner": Partner,
#         "Dependents": Dependents,
#         "PhoneService": PhoneService,
#         "MultipleLines": MultipleLines,
#         "InternetService": InternetService,
#         "OnlineSecurity": OnlineSecurity,
#         "OnlineBackup": OnlineBackup,
#         "DeviceProtection": DeviceProtection,
#         "TechSupport": TechSupport,
#         "StreamingTV": StreamingTV,
#         "StreamingMovies": StreamingMovies,
#         "Contract": Contract,
#         "PaperlessBilling": PaperlessBilling,
#         "PaymentMethod": PaymentMethod,
#         "tenure": int(tenure),
#         "MonthlyCharges": float(MonthlyCharges),
#         "TotalCharges": float(TotalCharges),
#     }
#     out = predict(payload)
#     return str(out)

# demo = gr.Interface(
#     fn=gradio_interface,
#     inputs=[
#         gr.Dropdown(["Male", "Female"], label="Gender"),
#         gr.Dropdown(["Yes", "No"], label="Partner"),
#         gr.Dropdown(["Yes", "No"], label="Dependents"),
#         gr.Dropdown(["Yes", "No"], label="Phone Service"),
#         gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines"),
#         gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
#         gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security"),
#         gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup"),
#         gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection"),
#         gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support"),
#         gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV"),
#         gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies"),
#         gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
#         gr.Dropdown(["Yes", "No"], label="Paperless Billing"),
#         gr.Dropdown(
#             ["Electronic check", "Mailed check",
#              "Bank transfer (automatic)", "Credit card (automatic)"],
#             label="Payment Method"
#         ),
#         gr.Number(label="Tenure (months)"),
#         gr.Number(label="Monthly Charges"),
#         gr.Number(label="Total Charges"),
#     ],
#     outputs="text",
#     title="Telco Churn Predictor",
#     description="Fill in the customer details to get a churn prediction.",
# )

# app = gr.mount_gradio_app(app, demo, path="/ui")

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from serving.inference import predict

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"status": "ok"}


class CustomerData(BaseModel):
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def api_predict(data: CustomerData):
    try:
        return {"prediction": predict(data.dict())}
    except Exception as e:
        return {"error": str(e)}


# ── Gradio prediction wrapper ─────────────────────────────────────────────────

def gradio_interface(
    gender, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges,
):
    payload = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }
    result = predict(payload)
    return str(result)


# ── Custom theme ──────────────────────────────────────────────────────────────

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("DM Sans"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("DM Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_sm,
    spacing_size=gr.themes.sizes.spacing_sm,
).set(
    # Backgrounds
    body_background_fill="#F8F9FA",
    block_background_fill="white",
    block_border_color="#E2E8F0",
    block_border_width="1px",
    # Header / title
    block_title_text_color="#1A202C",
    block_title_text_size="13px",
    block_title_text_weight="500",
    block_label_text_size="12px",
    block_label_text_color="#64748B",
    # Inputs
    input_background_fill="white",
    input_border_color="#E2E8F0",
    input_border_width="1px",
    input_shadow="none",
    input_placeholder_color="#94A3B8",
    # Button
    button_primary_background_fill="#0F6E56",
    button_primary_background_fill_hover="#085041",
    button_primary_text_color="white",
    button_primary_border_color="#0F6E56",
    button_secondary_background_fill="white",
    button_secondary_border_color="#E2E8F0",
)

# ── Custom CSS injected into Gradio ──────────────────────────────────────────

CUSTOM_CSS = """
/* Page wrapper */
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 24px 16px !important;
}

/* App title area */
#app-header {
    text-align: left;
    padding: 0 0 20px 0;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 24px;
}
#app-header h1 {
    font-size: 22px !important;
    font-weight: 600 !important;
    color: #F1F5F9 !important;
    margin: 0 0 4px 0 !important;
    letter-spacing: -0.3px;
}
#app-header p {
    font-size: 13px !important;
    color: #F1F5F9 !important;
    margin: 0 !important;
}

/* Section headers */
.section-header {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #94A3B8 !important;
    margin: 0 0 12px 0 !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid #F1F5F9 !important;
}

/* Block groups */
.block-group {
    background: white !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    padding: 16px !important;
    margin-bottom: 16px !important;
}

/* Dropdowns & numbers: tighter */
.gr-dropdown, .gr-number, select, input[type=number] {
    font-size: 13px !important;
    height: 36px !important;
    border-radius: 6px !important;
}

/* Labels: smaller, muted */
label span {
    font-size: 12px !important;
    color: #64748B !important;
    font-weight: 500 !important;
}

/* Output box */
#output-box textarea {
    font-family: 'DM Mono', monospace !important;
    font-size: 15px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    background: #1a1f2e !important;
    border: 1.5px solid #38BDF8 !important;
    min-height: 56px !important;
}

/* Submit button */
#predict-btn {
    background: #0F6E56 !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    letter-spacing: 0.01em;
}
#predict-btn:hover {
    background: #085041 !important;
}

/* Examples row */
.examples-header {
    font-size: 11px !important;
    color: #94A3B8 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* Remove Gradio footer branding */
footer { display: none !important; }
"""

# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="Churn Predictor") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
        <div id="app-header">
            <h1>Churn Predictor</h1>
            <p>XGBoost model · Telco customer churn risk assessment</p>
        </div>
    """)

    with gr.Row():

        # ── Left column ───────────────────────────────────────────────────────
        with gr.Column(scale=1):

            gr.HTML('<p class="section-header">Customer profile</p>')
            gender = gr.Dropdown(
                ["Male", "Female"], label="Gender", value="Male"
            )
            with gr.Row():
                Partner = gr.Dropdown(
                    ["Yes", "No"], label="Partner", value="No"
                )
                Dependents = gr.Dropdown(
                    ["Yes", "No"], label="Dependents", value="No"
                )

            gr.HTML('<p class="section-header" style="margin-top:16px">Phone service</p>')
            PhoneService = gr.Dropdown(
                ["Yes", "No"], label="Phone service", value="Yes"
            )
            MultipleLines = gr.Dropdown(
                ["Yes", "No", "No phone service"],
                label="Multiple lines",
                value="No",
            )

            gr.HTML('<p class="section-header" style="margin-top:16px">Billing & contract</p>')
            Contract = gr.Dropdown(
                ["Month-to-month", "One year", "Two year"],
                label="Contract type",
                value="Month-to-month",
            )
            PaperlessBilling = gr.Dropdown(
                ["Yes", "No"], label="Paperless billing", value="Yes"
            )
            PaymentMethod = gr.Dropdown(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                label="Payment method",
                value="Electronic check",
            )
            with gr.Row():
                tenure = gr.Number(
                    label="Tenure (months)", value=1, minimum=0, maximum=100, precision=0
                )
            with gr.Row():
                MonthlyCharges = gr.Number(
                    label="Monthly charges ($)", value=85.0, minimum=0, maximum=500
                )
                TotalCharges = gr.Number(
                    label="Total charges ($)", value=85.0, minimum=0, maximum=20000
                )

        # ── Right column ──────────────────────────────────────────────────────
        with gr.Column(scale=1):

            gr.HTML('<p class="section-header">Internet service</p>')
            InternetService = gr.Dropdown(
                ["DSL", "Fiber optic", "No"],
                label="Internet service",
                value="Fiber optic",
            )
            with gr.Row():
                OnlineSecurity = gr.Dropdown(
                    ["Yes", "No", "No internet service"],
                    label="Online security",
                    value="No",
                )
                OnlineBackup = gr.Dropdown(
                    ["Yes", "No", "No internet service"],
                    label="Online backup",
                    value="No",
                )
            with gr.Row():
                DeviceProtection = gr.Dropdown(
                    ["Yes", "No", "No internet service"],
                    label="Device protection",
                    value="No",
                )
                TechSupport = gr.Dropdown(
                    ["Yes", "No", "No internet service"],
                    label="Tech support",
                    value="No",
                )
            with gr.Row():
                StreamingTV = gr.Dropdown(
                    ["Yes", "No", "No internet service"],
                    label="Streaming TV",
                    value="Yes",
                )
                StreamingMovies = gr.Dropdown(
                    ["Yes", "No", "No internet service"],
                    label="Streaming movies",
                    value="Yes",
                )

            gr.HTML('<p class="section-header" style="margin-top:16px">Prediction</p>')
            output = gr.Textbox(
                label="Churn risk output",
                lines=2,
                interactive=False,
                elem_id="output-box",
                placeholder="Result will appear here after prediction…",
            )
            predict_btn = gr.Button(
                "Predict churn risk",
                variant="primary",
                elem_id="predict-btn",
            )

    # ── Examples ──────────────────────────────────────────────────────────────
    gr.HTML('<p class="examples-header" style="margin-top:8px; margin-bottom:6px;">Quick examples</p>')
    gr.Examples(
        examples=[
            [
                "Female", "No", "No", "Yes", "No",
                "Fiber optic", "No", "No", "No", "No", "Yes", "Yes",
                "Month-to-month", "Yes", "Electronic check",
                1, 85.0, 85.0,
            ],
            [
                "Male", "Yes", "Yes", "Yes", "Yes",
                "DSL", "Yes", "Yes", "Yes", "Yes", "No", "No",
                "Two year", "No", "Credit card (automatic)",
                60, 45.0, 2700.0,
            ],
        ],
        inputs=[
            gender, Partner, Dependents, PhoneService, MultipleLines,
            InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
            TechSupport, StreamingTV, StreamingMovies, Contract,
            PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges,
        ],
        label="",
    )

    # ── Event binding ─────────────────────────────────────────────────────────
    predict_btn.click(
        fn=gradio_interface,
        inputs=[
            gender, Partner, Dependents, PhoneService, MultipleLines,
            InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
            TechSupport, StreamingTV, StreamingMovies, Contract,
            PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges,
        ],
        outputs=output,
    )


# ── Mount Gradio into FastAPI ─────────────────────────────────────────────────
app = gr.mount_gradio_app(app, demo, path="/ui")