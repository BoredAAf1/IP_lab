# Import necessary libraries
# Pillow (PIL) is used for basic image input/output.
# NumPy is used for efficient array manipulation which is essential for image processing.
# Math is used for the square root function to calculate gradient magnitude.
from PIL import Image
import numpy as np
import math
import os

def convolve(image_array, kernel):
    """
    Performs a 2D convolution operation.

    Args:
        image_array (np.array): The input grayscale image as a NumPy array.
        kernel (np.array): The convolution kernel (e.g., Sobel operator).

    Returns:
        np.array: The resulting array after convolution.
    """
    # Get dimensions of the image and the kernel
    image_height, image_width = image_array.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate padding needed for the output image to have the same size as the input
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Create an empty array for the output with the same dimensions as the input image
    output_array = np.zeros_like(image_array)

    # Iterate over each pixel of the image
    for i in range(pad_height, image_height - pad_height):
        for j in range(pad_width, image_width - pad_width):
            # Extract the region of interest (ROI) from the image
            # This region has the same size as the kernel
            roi = image_array[i - pad_height : i + pad_height + 1, j - pad_width : j + pad_width + 1]
            
            # Apply the convolution: element-wise multiplication and sum
            output_pixel = np.sum(roi * kernel)
            output_array[i, j] = output_pixel
            
    return output_array

def edge_detection(image_path, operator_type='sobel'):
    """
    Performs edge detection on an image using the specified operator.

    Args:
        image_path (str): The path to the input image.
        operator_type (str): The operator to use ('sobel' or 'prewitt').

    Returns:
        Image: An image object with the detected edges.
    """
    try:
        # Open the image and convert it to grayscale
        print(f"Loading image from: {image_path}")
        img = Image.open(image_path).convert('L')
        # Convert the image to a NumPy array for processing
        img_array = np.array(img, dtype=np.float64)
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        print("Please make sure the image is in the same directory as the script.")
        return None

    # --- Define Kernels ---
    if operator_type.lower() == 'sobel':
        # Sobel Operator Kernels for detecting horizontal and vertical edges
        kernel_x = np.array([[-1, 0, 1], 
                             [-2, 0, 2], 
                             [-1, 0, 1]], dtype=np.float64)
        
        kernel_y = np.array([[-1, -2, -1], 
                             [ 0,  0,  0], 
                             [ 1,  2,  1]], dtype=np.float64)
        
    elif operator_type.lower() == 'prewitt':
        # Prewitt Operator Kernels
        kernel_x = np.array([[-1, 0, 1], 
                             [-1, 0, 1], 
                             [-1, 0, 1]], dtype=np.float64)
        
        kernel_y = np.array([[-1, -1, -1], 
                             [ 0,  0,  0], 
                             [ 1,  1,  1]], dtype=np.float64)
    else:
        raise ValueError("Unsupported operator type. Choose 'sobel' or 'prewitt'.")

    # --- Apply Convolution ---
    print(f"Applying {operator_type.capitalize()} operator...")
    # Convolve with the x-kernel to get the gradient in the x-direction (Gx)
    grad_x = convolve(img_array, kernel_x)
    # Convolve with the y-kernel to get the gradient in the y-direction (Gy)
    grad_y = convolve(img_array, kernel_y)

    # --- Calculate Gradient Magnitude ---
    print("Calculating gradient magnitude...")
    # Create an empty array for the magnitude
    magnitude = np.zeros_like(img_array)
    
    # Calculate the magnitude of the gradient using the formula: sqrt(Gx^2 + Gy^2)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            magnitude[i, j] = math.sqrt(grad_x[i, j]**2 + grad_y[i, j]**2)

    # --- Normalize the Output ---
    # The magnitude values can be outside the 0-255 range.
    # We scale them to fit within this range to create a visible image.
    magnitude_normalized = (magnitude / np.max(magnitude)) * 255
    
    # Convert the array back to an 8-bit unsigned integer format
    edge_img_array = magnitude_normalized.astype(np.uint8)
    
    # Create a new image from the resulting array
    edge_image = Image.fromarray(edge_img_array)
    
    return edge_image


if __name__ == "__main__":
    # Define the input image name
    input_image_name = "sample_image.jpg"
    
    # Check if the sample image exists before running
    if not os.path.exists(input_image_name):
        print(f"'{input_image_name}' not found. Creating a simple sample image.")
        # Create a simple black and white square image for demonstration
        sample_array = np.zeros((200, 200), dtype=np.uint8)
        sample_array[50:150, 50:150] = 255 # White square in the middle
        Image.fromarray(sample_array).save(input_image_name)
        print(f"'{input_image_name}' created successfully.")


    # --- Process with Sobel Operator ---
    sobel_edges = edge_detection(input_image_name, operator_type='sobel')
    if sobel_edges:
        sobel_output_path = "sobel_edges.jpg"
        sobel_edges.save(sobel_output_path)
        print(f"Sobel edge detection complete. Image saved as '{sobel_output_path}'")
        # sobel_edges.show() # Uncomment to display the image

    print("-" * 30)

    # --- Process with Prewitt Operator ---
    prewitt_edges = edge_detection(input_image_name, operator_type='prewitt')
    if prewitt_edges:
        prewitt_output_path = "prewitt_edges.jpg"
        prewitt_edges.save(prewitt_output_path)
        print(f"Prewitt edge detection complete. Image saved as '{prewitt_output_path}'")
        # prewitt_edges.show() # Uncomment to display the image
