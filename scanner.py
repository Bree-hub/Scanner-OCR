import cv2
import numpy as np
import imutils
from imutils import perspective
from rembg.bg import remove as rembg

# Constants
APPROX_POLY_DP_ACCURACY_RATIO = 0.05  # Less aggressive approximation
IMG_RESIZE_H = 500.0
PADDING = 10  # Add padding to the detected document edges

def scan_document(image_path):
    # Read image
    with open(image_path, 'rb') as f:
        data = f.read()

    # Process image using rembg for background removal
    bytes = np.frombuffer(rembg(data), np.uint8)
    img = cv2.imdecode(bytes, cv2.IMREAD_UNCHANGED)
    orig = img.copy()

    # Resize image
    ratio = img.shape[0] / IMG_RESIZE_H
    img = imutils.resize(img, height=int(IMG_RESIZE_H))

    # Convert image to binary format
    _, img = cv2.threshold(img[:, :, 3], 0, 255, cv2.THRESH_BINARY)

    # Apply median blur
    img = cv2.medianBlur(img, 15)

    # Find contours
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    outline = None

    # Iterate over contours to find a four-sided polygon
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        polygon = cv2.approxPolyDP(c, APPROX_POLY_DP_ACCURACY_RATIO * perimeter, True)

        if len(polygon) == 4:
            outline = polygon.reshape(4, 2)
            break  # Break after finding the first valid contour

    # If a valid outline is found, apply perspective transform
    if outline is not None:
        # Scale outline back to original image size
        outline = outline * ratio

        # Add padding to each point in the outline
        outline += PADDING

        # Ensure padding doesn't go out of bounds
        outline = np.clip(outline, 0, [orig.shape[1], orig.shape[0]])

        # Apply perspective transformation
        r = perspective.four_point_transform(orig, outline)
    else:
        # If no outline found, fallback to original or resized image
        r = orig

    # Save or display the result
    cv2.imwrite('scanned_document_with_padding.jpg', r)

    # Optionally return processed image for further use
    return r

if __name__ == "__main__":
    # Example usage
    image_path = 'uploads/car_logbook.jpeg'  
    scanned_image = scan_document(image_path)

    # Optionally display or further process scanned_image
    cv2.imshow('Scanned Document', scanned_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

