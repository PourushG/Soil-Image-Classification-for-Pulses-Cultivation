import cv2
import os

def preprocess_images(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over the input images
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Load the image
            image = cv2.imread(input_path)

            # Resize the image to a fixed size
            resized_image = cv2.resize(image, (256, 256))

            # Normalize pixel values to [0, 1]
            normalized_image = resized_image.astype(float) / 255.0

            # Convert the image to grayscale
            grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

            # Apply histogram equalization to the grayscale image
            equalized_image = cv2.equalizeHist(grayscale_image)

            # Save the preprocessed image
            cv2.imwrite(output_path, equalized_image)

            print(f"Preprocessed image saved: {filename}")

    print("Image preprocessing complete.")

# Example usage
input_directory = "C:/Users/pouru/Desktop/Augumented(C)"
output_directory = "C:/Users/pouru/Desktop/preprocessed(C)"

preprocess_images(input_directory, output_directory)
