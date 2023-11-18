import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
#import onnxruntime
import torch
import matplotlib.pyplot as plt
from segment_anything import predictor, sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

# Define the list to store clicked points
points = []
image = None

# Choose the model name
sam_model_name = 'vit_h'  

# Initialize the predictor
predictor = SamPredictor(sam_model_registry[sam_model_name]())

def show_mask(mask, ax, color=(30/255, 144/255, 255/255, 0.6)):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=facecolor, lw=lw))

def open_image():
    global image
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread('C:\\Users\\biswo\\Downloads\\biswojit_images\\biswojit_images\\64c0364025efe108d174aace.1692364756863.64c035fed879df3f08480eba.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.axis('on')
        plt.show()
        print("Image opened successfully")


def on_canvas_click(event):
    global image
    # Get the coordinates of the clicked point
    x, y = event.x, event.y
    points.append((x, y))

    # Segment and mask the clicked point
    segment_and_mask_point(x, y)
    def on_canvas_click(event):
    global image  # Add this line to indicate that you are using the global variable
    print("Clicked on canvas")  # Add this line for debugging
    if image is not None:
        # Get the coordinates of the clicked point
        x, y = event.x, event.y
        points.append((x, y))

        # Segment and mask the clicked point
        segment_and_mask_point(x, y)
    else:
        print("Please open an image first.")

def segment_and_mask_point(x, y):
    # Define the coordinates of the clicked point
    input_point = np.array([[x, y]])

    # Define a label for the point (e.g., 1 for object, 0 for background)
    input_label = np.array([1])

    # Create an array of coordinates with an additional (0, 0) point for background
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]

    # Create an array of labels with an additional -1 for background
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

    # Apply coordinate transformation for the clicked point
    onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

    # Create an input mask with zeros (you can customize this)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)

    # Create a flag indicating that a mask is provided
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    # Define the input data for the ONNX model
    ort_inputs = {
        "image_embeddings": None,  # Replace with the correct value
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
    }

    # Run the inference to obtain the mask
    # Replace ort_session.run with the correct way to run your ONNX model
    masks, _, low_res_logits = predictor.model.predict(**ort_inputs)

    # Convert the mask into a binary mask using the threshold
    masks = masks > predictor.model.mask_threshold

    # Display the image with the mask and clicked point
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks, plt.gca())          # Display the mask
    show_points(input_point, input_label, plt.gca())  # Mark the clicked point
    plt.axis('off')
    plt.show()



def main():
    global image

    root = tk.Tk()
    root.title("Interactive Segmentation")

    canvas = tk.Canvas(root, width=800, height=600)
    canvas.pack()
    canvas.bind("<Button-1>", on_canvas_click)

    open_button = tk.Button(root, text="Open Image", command=open_image)
    open_button.pack()

    root.mainloop()

if __name__ == "__main__":
    main()