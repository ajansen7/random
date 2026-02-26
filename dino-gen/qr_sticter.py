import cv2
import numpy as np
from pyzbar.pyzbar import decode

# Load your images (replace filenames with your actual file paths)
# 'piece.png' is the isolated QR fragment
# 'puzzle.jpg' is the main image with the hole
piece = cv2.imread('Screenshot_20251222-175703.png')
puzzle = cv2.imread('PXL_20251223_005656289.jpg')

# 1. CROP THE PIECE
# We need to isolate just the QR sticker from the first image
# Converting to grayscale and thresholding to find the sticker
gray_piece = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_piece, 200, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assume the largest contour is the sticker
if contours:
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    sticker_crop = piece[y:y+h, x:x+w]
else:
    # Fallback if detection fails
    sticker_crop = piece

# ... (After loading 'piece' and 'puzzle') ...

# Resize piece to a reasonable guess (or keep your current 140 if it looks right)
target_size = 140
resized_piece = cv2.resize(sticker_crop, (target_size, target_size))

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: {x}, {y}")
        
        # Make a copy so we don't mess up the original
        temp_img = puzzle.copy()
        
        # Paste the piece centered on where you clicked
        y_start = y - target_size // 2
        x_start = x - target_size // 2
        
        # Safety bounds check
        if y_start >= 0 and x_start >= 0:
            try:
                temp_img[y_start:y_start+target_size, x_start:x_start+target_size] = resized_piece
                cv2.imshow('Click to Place', temp_img)
                
                # Try decoding immediately
                decoded = decode(temp_img)
                if decoded:
                    print(f"\nSUCCESS! URL: {decoded[0].data.decode('utf-8')}")
            except ValueError:
                print("Clicked too close to the edge!")

print("CLICK on the center of the missing hole in the window that opens.")
cv2.imshow('Click to Place', puzzle)
cv2.setMouseCallback('Click to Place', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. DECODE
decoded_objects = decode(combined)

if decoded_objects:
    for obj in decoded_objects:
        print("FOUND URL:", obj.data.decode("utf-8"))
else:
    print("Could not decode. Try adjusting the x_offset/y_offset or target_size.")
    # Save the image to see how it looks
    cv2.imwrite('debug_combined.jpg', combined)