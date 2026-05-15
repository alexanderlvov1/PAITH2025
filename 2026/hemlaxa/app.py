# This file is based on streamlit-mnist-drawable by Rahul Sharma.
# Source: https://github.com/rahulsrma26/streamlit-mnist-drawable
# Forked on: 2026-05-02
# Changes from original: minimal (small adjustments for integration).

import os
import joblib
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas

MODEL_PATH_LR = os.path.join(os.path.dirname(__file__), 'mnist_lr_model.joblib')
MODEL_PATH_SVC = os.path.join(os.path.dirname(__file__), 'mnist_svc_model.joblib')
MODEL_PATH_CNN = os.path.join(os.path.dirname(__file__), 'mnist_cnn_model.keras')
#MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.keras')

if not (os.path.isfile(MODEL_PATH_CNN)):
    st.error('CNN model was not found. Train the model first and make sure the path is correct (see README.md).')
    st.stop()

if not (os.path.isfile(MODEL_PATH_LR)):
    st.error('LR model was not found. Train the model first and make sure the path is correct (see README.md).')
    st.stop()

if not (os.path.isfile(MODEL_PATH_SVC)):
    st.error('SVC model was not found. Train the model first and make sure the path is correct (see README.md).')
    st.stop()

model_CNN = load_model(MODEL_PATH_CNN)
model_LR = joblib.load(MODEL_PATH_LR)
model_SVC = joblib.load(MODEL_PATH_SVC)
# st.markdown('<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

st.title('My Digit Recognizer')
st.markdown('Draw a digit and hit **Predict** to run it through all three models!')

# Display size of the drawing canvas in pixels — purely cosmetic, can be adjusted freely.
# The drawn image is always downscaled to 28x28 by preprocess() regardless of this value.
SIZE = 192
mode = st.checkbox("Draw (or Delete)?", True)
drawing_mode = "freedraw" if mode else "transform"

canvas = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode=drawing_mode,
    key='canvas')



def preprocess(canvas_data):
    """Convert a canvas image into an MNIST-like 28x28 normalized digit."""
    rgba = canvas_data.astype('uint8')
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    # Threshold to strict black/white — removes anti-aliasing noise from the canvas stroke
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(binary)
    if coords is None:
        return None

    # Crop tightly around the drawn pixels so the digit fills the frame
    x, y, width, height = cv2.boundingRect(coords)
    digit = binary[y:y + height, x:x + width]

    # Embed into a square canvas to preserve aspect ratio before resizing
    side = max(width, height)
    square = np.zeros((side, side), dtype=np.uint8)
    x_offset = (side - width) // 2
    y_offset = (side - height) // 2
    square[y_offset:y_offset + height, x_offset:x_offset + width] = digit

    # INTER_AREA averages pixel blocks when downscaling — best quality for shrinking,
    # avoids aliasing artifacts that INTER_LINEAR or INTER_NEAREST would introduce
    resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
    # MNIST digits are 20x20 centered in a 28x28 frame with a 4-pixel border on each side
    padded = np.pad(resized, ((4, 4), (4, 4)), mode='constant')
    normalized = padded.astype('float32') / 255.0
    return normalized


def show_prediction(canvas_result, label, model, model_name):
    if canvas_result.image_data is not None:
        img28 = preprocess(canvas_result.image_data)
        if img28 is None:
            st.warning(f'Draw {label} first.')
            return

        # INTER_NEAREST keeps hard pixel edges when upscaling — shows the exact
        # 28x28 input the model sees without blurring introduced by smoother methods
        rescaled = cv2.resize(img28, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        st.write(f'Model Input ({model_name})')
        st.image(rescaled, clamp=True)

        from tensorflow.keras import Model as KerasModel
        if isinstance(model, KerasModel):
            probs = model.predict(img28.reshape(1, 28, 28, 1), verbose=0)[0]
            # CNN outputs a probability for each of the 10 classes (softmax);
            # argmax returns the index (= digit) with the highest probability
            pred_class = np.argmax(probs)
        else:
            flat = img28.reshape(1, -1)
            pred_class = model.predict(flat)[0]
            # SVC only supports predict_proba() if trained with probability=True;
            # hasattr() is unreliable here because the method always exists on the class
            # but raises at runtime when probability=False (the sklearn default).
            prob_available = getattr(model, 'probability', True)
            probs = model.predict_proba(flat)[0] if prob_available else None

        st.write(f'**{label} → {pred_class}**')
        values = probs if probs is not None else np.zeros(10, dtype='float32')
        st.bar_chart(pd.Series(values, index=list(range(10)), name='probability'))
    else:
        st.warning(f'Draw {label} first.')


if st.button('Predict'):
    res1, res2, res3 = st.columns(3)
    with res1:
        show_prediction(canvas, "CNN", model_CNN, "CNN")
    with res2:
        show_prediction(canvas, "LR", model_LR, "LR")
    with res3:
        show_prediction(canvas, "SVC", model_SVC, "SVC")
