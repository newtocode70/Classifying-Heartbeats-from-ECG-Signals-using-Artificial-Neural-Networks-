from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

# Add the project root so this file can import src.model when run with Streamlit.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ECGNet

MODEL_PATH = PROJECT_ROOT / "results" / "best_model.pth"
CLASS_NAMES = [
    "N - Normal",
    "S - Supraventricular ectopic",
    "V - Ventricular ectopic",
    "F - Fusion",
    "Q - Unknown / paced"
]

st.set_page_config(page_title="ECG Heartbeat Classifier", page_icon="❤️")
st.title("ECG Heartbeat Classification")
st.write(
    "Upload one ECG heartbeat containing 187 signal values, "
    "or choose a row from the provided test dataset."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python -m src.train` first."
        )
    model = ECGNet()
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def predict_signal(signal):
    x = torch.tensor(signal, dtype=torch.float32).reshape(1, 187)
    model = load_model()
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class = int(torch.argmax(probabilities).item())
    return predicted_class, probabilities.numpy()

uploaded_file = st.file_uploader(
    "Upload CSV file (187 signal columns; optional 188th label column)",
    type=["csv"]
)

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file, header=None)
    if uploaded_df.shape[1] == 188:
        st.info("Detected 188 columns. The last column is treated as the label and ignored.")
        uploaded_df = uploaded_df.iloc[:, :187]

    if uploaded_df.shape[1] != 187:
        st.error(
            f"Expected 187 signal columns, but received {uploaded_df.shape[1]}."
        )
        st.stop()

    row_number = st.number_input(
        "Select uploaded heartbeat row",
        min_value=0,
        max_value=len(uploaded_df) - 1,
        value=0,
        step=1
    )
    signal = uploaded_df.iloc[int(row_number)].astype(float).to_numpy()

else:
    st.caption(
        "Alternatively, place mitbih_test.csv in the project's data folder "
        "and select one of its rows below."
    )
    test_path = PROJECT_ROOT / "data" / "mitbih_test.csv"

    if not test_path.exists():
        st.warning(
            "No test CSV found. Upload a CSV above, or place "
            "mitbih_test.csv in the data folder."
        )
        st.stop()

    test_df = pd.read_csv(test_path, header=None)
    row_number = st.number_input(
        "Select test dataset row",
        min_value=0,
        max_value=len(test_df) - 1,
        value=0,
        step=1
    )
    signal = test_df.iloc[int(row_number), :187].astype(float).to_numpy()

if st.button("Classify heartbeat", type="primary"):
    try:
        predicted_class, probabilities = predict_signal(signal)

        st.subheader("Prediction")
        st.success(f"Predicted class: {CLASS_NAMES[predicted_class]}")

        fig, ax = plt.subplots(figsize=(10, 3))
        time_ms = [1000 * i / 125 for i in range(187)]
        ax.plot(time_ms, signal)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Normalised amplitude")
        ax.set_title("ECG Heartbeat Signal")
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Class probabilities")
        probability_df = pd.DataFrame({
            "Heartbeat class": CLASS_NAMES,
            "Probability": probabilities,
            "Probability (%)": probabilities * 100
        })
        st.dataframe(probability_df, use_container_width=True)
        st.bar_chart(
            probability_df.set_index("Heartbeat class")["Probability"]
        )

    except Exception as error:
        st.error(f"Could not classify this heartbeat: {error}")
