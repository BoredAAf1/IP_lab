from PIL import Image
import numpy as np

def create_gaussian_low_pass_filter(shape, cutoff_frequency):
    """
    Creates a 2D Gaussian low-pass filter.

    Args:
        shape (tuple): The size of the filter, e.g., (height, width).
        cutoff_frequency (float): The cutoff frequency of the filter.
                                  Higher values allow more high-frequency content to pass.
    
    Returns:
        numpy.ndarray: The 2D Gaussian low-pass filter.
    """
    h, w = shape
    center_y, center_x = h // 2, w // 2

    y = np.arange(h)
    x = np.arange(w)

    y = y.reshape(h, 1)
    x = x.reshape(1, w)

    distance_squared = (y - center_y)**2 + (x - center_x)**2

    filter_array = np.exp(-distance_squared / (2 * cutoff_frequency**2))

    return filter_array

def process_channel(channel_data, filter_array):
    """
    Processes a single color channel (R, G, or B) in the frequency domain.

    Args:
        channel_data (numpy.ndarray): The 2D array of pixel values for one channel.
        filter_array (numpy.ndarray): The 2D filter to be applied.

    Returns:
        numpy.ndarray: The filtered channel data in the spatial domain.
    """
    # Perform 2D Fast Fourier Transform (FFT)
    fourier_transform = np.fft.fft2(channel_data)

    # Shift the zero-frequency component to the center of the spectrum
    shifted_fourier_transform = np.fft.fftshift(fourier_transform)

    # Apply the filter by element-wise multiplication
    filtered_fourier_transform = shifted_fourier_transform * filter_array

    # Shift the zero-frequency component back to its original position
    shifted_back_fourier_transform = np.fft.ifftshift(filtered_fourier_transform)

    # Perform the Inverse 2D FFT
    inverse_fourier_transform = np.fft.ifft2(shifted_back_fourier_transform)

    # Get the real part of the result and clamp values to a valid pixel range (0-255).
    filtered_channel = np.real(inverse_fourier_transform)
    
    # Scale and clip the values to be within the 0-255 range for image saving.
    filtered_channel = np.clip(filtered_channel, 0, 255)

    return filtered_channel.astype(np.uint8)

def main():
    """
    Main function to orchestrate the image filtering process.
    """
    try:
        # Load the color image
        image_path = "sample_image.jpg"
        original_image = Image.open(image_path)
        print(f"Image '{image_path}' loaded successfully. [Image of a racing car on a track] ")
        
        # Convert the image to a NumPy array
        img_array = np.array(original_image)
        h, w, c = img_array.shape

        # Define filter parameters
        # A smaller cutoff_frequency value results in more blurring
        low_pass_cutoff = 50
        high_pass_cutoff = 20
        band_pass_low_cutoff = 10
        band_pass_high_cutoff = 60

        # Create the filters
        low_pass_filter = create_gaussian_low_pass_filter((h, w), low_pass_cutoff)
        
        # High-pass filter is the inverse of a low-pass filter
        high_pass_filter = 1 - create_gaussian_low_pass_filter((h, w), high_pass_cutoff)
        
        # Band-pass filter is the difference between two low-pass filters
        band_pass_filter = create_gaussian_low_pass_filter((h, w), band_pass_high_cutoff) - \
                           create_gaussian_low_pass_filter((h, w), band_pass_low_cutoff)

        print("All three filters (Low-pass, High-pass, Band-pass) created.")

        # Split the image into its color channels (Red, Green, Blue)
        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1]
        blue_channel = img_array[:, :, 2]

        # Process each channel with the LOW-PASS filter
        print("Processing with Low-pass filter...")
        filtered_red_lp = process_channel(red_channel, low_pass_filter)
        filtered_green_lp = process_channel(green_channel, low_pass_filter)
        filtered_blue_lp = process_channel(blue_channel, low_pass_filter)
        filtered_img_lp = np.stack([filtered_red_lp, filtered_green_lp, filtered_blue_lp], axis=-1)
        Image.fromarray(filtered_img_lp).save("filtered_image.png")
        print("Low-pass filtered image saved to 'filtered_image.png'.")

        # Process each channel with the HIGH-PASS filter
        print("Processing with High-pass filter...")
        filtered_red_hp = process_channel(red_channel, high_pass_filter)
        filtered_green_hp = process_channel(green_channel, high_pass_filter)
        filtered_blue_hp = process_channel(blue_channel, high_pass_filter)
        filtered_img_hp = np.stack([filtered_red_hp, filtered_green_hp, filtered_blue_hp], axis=-1)
        Image.fromarray(filtered_img_hp).save("high_pass_filtered_image.png")
        print("High-pass filtered image saved to 'high_pass_filtered_image.png'.")

        # Process each channel with the BAND-PASS filter
        print("Processing with Band-pass filter...")
        filtered_red_bp = process_channel(red_channel, band_pass_filter)
        filtered_green_bp = process_channel(green_channel, band_pass_filter)
        filtered_blue_bp = process_channel(blue_channel, band_pass_filter)
        filtered_img_bp = np.stack([filtered_red_bp, filtered_green_bp, filtered_blue_bp], axis=-1)
        Image.fromarray(filtered_img_bp).save("band_pass_filtered_image.png")
        print("Band-pass filtered image saved to 'band_pass_filtered_image.png'.")

    except FileNotFoundError:
        print(f"Error: The image file '{image_path}' was not found.")
        print("Please make sure 'sample_image.jpg' is in the same directory as the script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
