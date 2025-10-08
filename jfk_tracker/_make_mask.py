import cv2
import numpy as np
import os

# Configuration
video_name = "Asheque Rahman - Export_ T4 _ Arrivals _ 1.28.2025 _ 3 pm - 4 pm"
num_lanes = 9  # Number of lanes to create masks for
base_dir = r"c:\Users\leo\Desktop\jfk"
image_path = os.path.join(base_dir, "masks", "ref_imgs", f"{video_name}_colored_ref.jpg")

# Create output directory
output_dir = os.path.join(base_dir, "masks", f"masks_{video_name}")
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Load the image
image = cv2.imread(image_path)
if image is None:
    print(f"Failed to load image: {image_path}")
    exit()

# Initialize variables
current_lane = 1
color_selected = None
mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Mask with the same size as the image, single channel

def click_event(event, x, y, flags, param):
    global color_selected, mask, current_lane

    # First click: select color
    if event == cv2.EVENT_LBUTTONDOWN and color_selected is None:
        color_selected = image[y, x].tolist()  # Get the BGR color at the clicked point
        # color_selected = [255,111,0]
        print(f"Lane {current_lane}/{num_lanes} - Color selected at ({x}, {y}): {color_selected}")
        
    # Second click: create binary mask and flood fill within the selected color's area
    elif event == cv2.EVENT_LBUTTONDOWN and color_selected is not None:
        print(f"Lane {current_lane}/{num_lanes} - Filling area starting at ({x}, {y}) with selected color {color_selected}")
        
        # Define tolerance for color range
        tolerance = 15  # Reduced from 40 to prevent color bleeding
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
        
        print(f"Lane {current_lane}/{num_lanes} - Area filled and mask created.")

        # Show the mask for preview
        cv2.imshow("Mask", mask)
        
        # Save the mask immediately
        mask_output_path = os.path.join(output_dir, f"mask_{current_lane}.png")
        cv2.imwrite(mask_output_path, mask)
        print(f"Lane {current_lane}/{num_lanes} - Mask saved to {mask_output_path}")
        
        # Reset for next lane
        color_selected = None
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        current_lane += 1
        
        if current_lane <= num_lanes:
            print(f"\n--- Ready for Lane {current_lane}/{num_lanes} ---")
            print("Click to select color, then click to fill area")
        else:
            print(f"\n=== All {num_lanes} lanes completed! ===")
            print(f"All masks saved to: {output_dir}")
            print("Press any key to exit")

# Display the image and set the click event
print(f"Creating masks for: {video_name}")
print(f"--- Ready for Lane {current_lane}/{num_lanes} ---")
print("Click to select color, then click to fill area")
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", click_event)

# Wait until the user presses a key to close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"\nAll masks saved to: {output_dir}")