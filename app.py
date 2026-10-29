import os
import gradio as gr
import numpy as np
from PIL import Image, ImageOps


script_dir = os.path.dirname(os.path.abspath(__file__))

w1 = np.load(os.path.join(script_dir, "w1.npy"))
b1 = np.load(os.path.join(script_dir, "b1.npy"))
w2 = np.load(os.path.join(script_dir, "w2.npy"))
b2 = np.load(os.path.join(script_dir, "b2.npy"))


# Activation functions
def relu(z):
    return np.maximum(0, z)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def center_of_mass_shift(img28):
    total = np.sum(img28)
    if total == 0:
        return img28


    cy = np.sum(np.arange(28)[:, None] * img28) / total
    cx = np.sum(np.arange(28)[None, :] * img28) / total

    # Target center is (14, 14)
    shift_y = int(round(14 - cy)) 
    shift_x = int(round(14 - cx))

    shifted = np.zeros_like(img28)
    src_y_start = max(0, -shift_y)
    src_y_end = min(28, 28 - shift_y)
    dst_y_start = max(0, shift_y)
    dst_y_end = min(28, 28 + shift_y)

    src_x_start = max(0, -shift_x)
    src_x_end = min(28, 28 - shift_x)
    dst_x_start = max(0, shift_x)
    dst_x_end = min(28, 28 + shift_x)

    shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = img28[
        src_y_start:src_y_end, src_x_start:src_x_end
    ]
    return shifted


# MNIST Preprocessing Pipeline
def preprocess_canvas(canvas_input):
    if canvas_input is None:
        return None, None

    #handling different image formats (dict, ndarray, or file path)
    if isinstance(canvas_input, dict):
        image = canvas_input.get("composite", None)
        if image is None:
            layers = canvas_input.get("layers", [])
            image = layers[0] if len(layers) > 0 else canvas_input.get("background")
    else:
        image = canvas_input

    if image is None:
        return None, None

    # Convert to PIL Image
    if isinstance(image, str):
        pil_img = Image.open(image)
    elif isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image.astype(np.uint8))
    else:
        return None, None

    # Convert to grayscale
    pil_img = pil_img.convert("L")

   
    img_array = np.array(pil_img)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

   
    img_array = np.where(img_array < 45, 0, img_array)

    
    coords = np.argwhere(img_array > 0)
    if coords.size == 0:
        return None, None  # Empty canvas

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

  
    cropped = img_array[y_min : y_max + 1, x_min : x_max + 1]
    cropped_pil = Image.fromarray(cropped.astype(np.uint8))

  
    w, h = cropped_pil.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round((h * 20.0) / w)))
    else:
        new_h = 20
        new_w = max(1, int(round((w * 20.0) / h)))

    resized = np.array(
        cropped_pil.resize((new_w, new_h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )

  
    temp_canvas = np.zeros((28, 28), dtype=np.float32)
    paste_y = (28 - new_h) // 2
    paste_x = (28 - new_w) // 2
    temp_canvas[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = resized

  
    final_canvas = center_of_mass_shift(temp_canvas)

   
    x = final_canvas.reshape(1, 784) / 255.0
    preview_img = final_canvas.astype(np.uint8)

    return x, preview_img


#Predict function
def predict(canvas_data):
    x, preview_28x28 = preprocess_canvas(canvas_data)

    if x is None:
        # User hasn't drawn anything or cleared canvas
        blank_preview = np.zeros((28, 28), dtype=np.uint8)
        neutral_probs = {str(i): 0.1 for i in range(10)}
        return neutral_probs, blank_preview

    # Pure NumPy Forward Pass
    z1 = np.dot(x, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)

    probabilities = a2[0]
    result_dict = {str(i): float(probabilities[i]) for i in range(10)}

    return result_dict, preview_28x28


#Gradio UI Interface
title = "Pure NumPy MNIST Digit Recognizer"
description = (
    "Draw any single digit (0–9) on the canvas. "
    "This model was built from scratch using **pure NumPy matrix math** — "
    "**no PyTorch, no TensorFlow** for inference! "
    "The 28x28 box shows the exact Center-of-Mass preprocessed image the neural network sees."
)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(label="Draw a Digit (0-9)"),
    outputs=[
        gr.Label(num_top_classes=3, label="Top Predictions"),
        gr.Image(label="What Model Sees (28x28 Center of Mass)", width=140, height=140),
    ],
    title=title,
    description=description,
    flagging_mode="never",  
)

if __name__ == "__main__":
    demo.launch()