import cv2
import numpy as np

# Load the image
image_path = "image.png"
image = cv2.imread(image_path)
if image is None:
    print("Failed to load image.")
    exit()

# Initialize variables
color_selected = None
mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Mask with the same size as the image, single channel

def click_event(event, x, y, flags, param):
    global color_selected, mask

    # First click: select color
    if event == cv2.EVENT_LBUTTONDOWN and color_selected is None:
        color_selected = image[y, x].tolist()  # Get the BGR color at the clicked point
        # color_selected = [255,111,0]
        print(f"Color selected at ({x}, {y}): {color_selected}")
        
    # Second click: create binary mask and flood fill within the selected color's area
    elif event == cv2.EVENT_LBUTTONDOWN and color_selected is not None:
        print(f"Filling area starting at ({x}, {y}) with selected color {color_selected}")
        
        # Define tolerance for color range
        tolerance = 40  # Adjust tolerance as needed
        lower_bound = np.clip(np.array(color_selected) - tolerance, 0, 255)
        upper_bound = np.clip(np.array(color_selected) + tolerance, 0, 255)
        
        # Convert image to binary based on the selected color
        binary_image = cv2.inRange(image, lower_bound, upper_bound)
        binary_image = cv2.bitwise_not(binary_image)  # Invert so selected color is black, others white

        # Show the binary image for reference
        cv2.imshow("Binary Image", binary_image)

        # Create a flood fill mask
        flood_fill_img = binary_image.copy()
        flood_mask = np.zeros((binary_image.shape[0] + 2, binary_image.shape[1] + 2), dtype=np.uint8)
        
        # Apply flood fill on the binary image, starting at the second click location
        floodFillFlags = cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        cv2.floodFill(flood_fill_img, flood_mask, (x, y), 255, flags=floodFillFlags)
        
        # Convert flood_mask to a usable mask format (ignoring the border padding)
        mask = flood_mask[1:-1, 1:-1]
        mask[mask > 0] = 255  # Set the filled area to white, everything else remains black
        
        print("Area filled and mask created.")

        # Show the mask for preview
        cv2.imshow("Mask", mask)

# Display the image and set the click event
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", click_event)

# Wait until the user presses a key to close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the mask
mask_output_path = "mask.png"
cv2.imwrite(mask_output_path, mask)
print(f"Mask saved to {mask_output_path}")
