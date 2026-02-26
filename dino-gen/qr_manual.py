import cv2
import numpy as np
from pyzbar.pyzbar import decode

# FILE PATHS - Check these match your files
MAIN_IMAGE_PATH = 'PXL_20251223_005656289.jpg'  # The drawing with the hole
PIECE_IMAGE_PATH = 'Screenshot_20251222-175703.png' # The sticker piece

def stitch_and_solve():
    # 1. Load Images
    main_img = cv2.imread(MAIN_IMAGE_PATH)
    piece_img = cv2.imread(PIECE_IMAGE_PATH)

    if main_img is None or piece_img is None:
        print("Error: Could not find images. Check your file paths!")
        return

    print("Step 1: Draw a box around the EMPTY HOLE in the main image.")
    print("        Press SPACE or ENTER to confirm selection.")
    
    # Allows you to drag a rectangle around the hole
    # Returns (x, y, w, h)
    hole_rect = cv2.selectROI("Select the HOLE", main_img, showCrosshair=True)
    cv2.destroyWindow("Select the HOLE")
    
    # Check if user cancelled
    if hole_rect[2] == 0 or hole_rect[3] == 0:
        print("No selection made. Exiting.")
        return

    print("Step 2: Draw a box around the PUZZLE PIECE in the second image.")
    print("        Try to crop it tightly (exclude whitespace).")
    
    piece_rect = cv2.selectROI("Select the PIECE", piece_img, showCrosshair=True)
    cv2.destroyWindow("Select the PIECE")

    if piece_rect[2] == 0 or piece_rect[3] == 0:
        print("No selection made. Exiting.")
        return

    # 2. Process: Crop and Resize
    # Unpack coordinates
    (hx, hy, hw, hh) = hole_rect
    (px, py, pw, ph) = piece_rect

    # Crop the piece image to just the selection
    piece_crop = piece_img[py:py+ph, px:px+pw]

    # Resize the cropped piece to match the dimensions of the hole exactly
    piece_resized = cv2.resize(piece_crop, (hw, hh))

    # 3. Stitch
    combined = main_img.copy()
    # Paste the resized piece into the hole location
    combined[hy:hy+hh, hx:hx+hw] = piece_resized

    # 4. Show Result
    print("Stitching complete. Attempting to decode...")
    cv2.imshow("Result (Press key to close)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 5. Decode
    decoded_objects = decode(combined)
    
    if decoded_objects:
        print("\nSUCCESS! found the QR Code data:")
        print("------------------------------------------------")
        for obj in decoded_objects:
            print(obj.data.decode("utf-8"))
        print("------------------------------------------------")
    else:
        print("\nCould not decode automatically.")
        print("Tip: If the result image looked misaligned, try running again and")
        print("being more precise with your rectangle selection.")
        # Save it so you can scan it with your phone if you want
        cv2.imwrite("solved_puzzle.jpg", combined)
        print("Saved 'solved_puzzle.jpg' - try scanning this file with your phone.")

if __name__ == "__main__":
    stitch_and_solve()