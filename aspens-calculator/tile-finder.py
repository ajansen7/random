import cv2
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================

ASPECT_RATIO = 1.28

# Default Colors (HSV)
DEF_PINE_HUE = 70
DEF_PINE_TOL = 25
DEF_ASPEN_HUE = 20
DEF_ASPEN_TOL = 15

# ==========================================
# HELPERS
# ==========================================

def nothing(x): pass

input_points = []
def mouse_handler(event, x, y, flags, param):
    global input_points
    if len(input_points) < 4 and event == cv2.EVENT_LBUTTONDOWN:
        input_points.append([x, y])

def get_diamond_warp(image, pts):
    target_h = 1000
    target_w = int(target_h * ASPECT_RATIO)
    pad_x = target_w * 0.1
    pad_y = target_h * 0.1
    
    dst_top = [target_w / 2, pad_y]
    dst_right = [target_w - pad_x, target_h / 2]
    dst_bottom = [target_w / 2, target_h - pad_y]
    dst_left = [pad_x, target_h / 2]
    
    src_arr = np.array(pts, dtype="float32")
    dst_arr = np.array([dst_top, dst_right, dst_bottom, dst_left], dtype="float32")

    M = cv2.getPerspectiveTransform(src_arr, dst_arr)
    warped = cv2.warpPerspective(image, M, (target_w, target_h))
    return warped

# ==========================================
# MAIN
# ==========================================

def process_aspens(image_path):
    global input_points
    
    img = cv2.imread(image_path)
    if img is None: 
        print("Error loading image")
        return

    # 1. WARP
    scale = 1000 / img.shape[0]
    disp_h = 1000
    disp_w = int(img.shape[1] * scale)
    disp = cv2.resize(img, (disp_w, disp_h))
    
    print("\nSTEP 1: Click 4 'COMPASS POINTS' (Top, Right, Bottom, Left)")
    cv2.namedWindow("Setup")
    cv2.setMouseCallback("Setup", mouse_handler)
    labels = ["TOP", "RIGHT", "BOT", "LEFT"]
    
    while len(input_points) < 4:
        temp = disp.copy()
        for i, pt in enumerate(input_points):
            cv2.circle(temp, tuple(pt), 6, (0, 255, 0), -1)
            cv2.putText(temp, labels[i], (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Setup", temp)
        if cv2.waitKey(20) == 27: return

    pts_full = np.array(input_points, dtype=np.float32) * (1/scale)
    warped = get_diamond_warp(img, pts_full)
    cv2.destroyAllWindows()

    # 2. AGGRESSIVE TILE DETECTION
    print("\nSTEP 2: TUNE BORDERS")
    print("1. Increase 'Line Width' until borders pop out.")
    print("2. Adjust 'Sensitivity' to clear the noise.")
    
    cv2.namedWindow("Tile Tuner")
    # Kernel size for TopHat (odd numbers 1-31)
    cv2.createTrackbar("Line Width", "Tile Tuner", 15, 30, nothing) 
    # Adaptive Threshold C constant
    cv2.createTrackbar("Sensitivity", "Tile Tuner", 10, 50, nothing)
    
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This acts like a "local" contrast booster
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_enh = clahe.apply(gray)
    
    valid_tile_centers = []

    while True:
        k_size = cv2.getTrackbarPos("Line Width", "Tile Tuner")
        sens = cv2.getTrackbarPos("Sensitivity", "Tile Tuner")
        
        # Ensure odd kernel size
        if k_size % 2 == 0: k_size += 1
        if k_size < 3: k_size = 3
        
        # 1. TOP-HAT FILTER
        # Subtracts the local background, leaving only bright features smaller than k_size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        tophat = cv2.morphologyEx(gray_enh, cv2.MORPH_TOPHAT, kernel)
        
        # 2. NORMALIZE
        # Stretch the contrast of the tophat result to full dynamic range (0-255)
        tophat_norm = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
        
        # 3. OTSU THRESHOLDING
        # Automatically finds the best cutoff for the "brightest" parts
        _, binary = cv2.threshold(tophat_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. CLEANUP
        # Remove tiny noise specs
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        # Connect broken lines
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        
        # Find Contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        preview = warped.copy()
        valid_tile_centers = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300 or area > 6000: continue
            
            # Circularity Check
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.6: continue 
            
            (x,y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            
            # Draw Found Tile
            cv2.circle(preview, center, int(radius), (255, 0, 0), 2) # Blue Ring
            valid_tile_centers.append((center, int(radius)))

        # Show the intermediate "Mask" so you can debug the filter
        # Resize mask to overlay in corner
        mask_mini = cv2.cvtColor(cv2.resize(binary, (300, 300)), cv2.COLOR_GRAY2BGR)
        preview[0:300, 0:300] = mask_mini
        cv2.putText(preview, "Mask View", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.putText(preview, f"Tiles Found: {len(valid_tile_centers)}", (320, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Tile Tuner", cv2.resize(preview, (1000, 800)))
        if cv2.waitKey(20) == 13: break 

    cv2.destroyAllWindows()

    # 3. TREE DETECTION (Blob Analysis)
    print("\nSTEP 3: TUNE TREE COLORS")
    cv2.namedWindow("Tree Tuner")
    cv2.createTrackbar("P-Hue", "Tree Tuner", DEF_PINE_HUE, 179, nothing)
    cv2.createTrackbar("P-Tol", "Tree Tuner", DEF_PINE_TOL, 50, nothing)
    cv2.createTrackbar("A-Hue", "Tree Tuner", DEF_ASPEN_HUE, 179, nothing)
    cv2.createTrackbar("A-Tol", "Tree Tuner", DEF_ASPEN_TOL, 50, nothing)
    
    blurred = cv2.GaussianBlur(warped, (9, 9), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    final_pines = 0
    final_aspens = 0

    while True:
        p_h = cv2.getTrackbarPos("P-Hue", "Tree Tuner")
        p_t = cv2.getTrackbarPos("P-Tol", "Tree Tuner")
        a_h = cv2.getTrackbarPos("A-Hue", "Tree Tuner")
        a_t = cv2.getTrackbarPos("A-Tol", "Tree Tuner")
        
        # Color Masks
        lower_p = np.array([max(0, p_h - p_t), 50, 50])
        upper_p = np.array([min(179, p_h + p_t), 255, 255])
        mask_p = cv2.inRange(hsv, lower_p, upper_p)
        
        lower_a = np.array([max(0, a_h - a_t), 80, 80])
        upper_a = np.array([min(179, a_h + a_t), 255, 255])
        mask_a = cv2.inRange(hsv, lower_a, upper_a)
        
        preview = warped.copy()
        p_count = 0
        a_count = 0
        
        # Check detected tiles
        for (center, rad) in valid_tile_centers:
            cx, cy = center
            check_rad = int(rad * 0.6)
            
            y1, y2 = max(0, cy-check_rad), min(1000, cy+check_rad)
            x1, x2 = max(0, cx-check_rad), min(preview.shape[1], cx+check_rad)
            
            p_area = cv2.countNonZero(mask_p[y1:y2, x1:x2])
            a_area = cv2.countNonZero(mask_a[y1:y2, x1:x2])
            
            limit = 20
            
            if p_area > limit and p_area > a_area:
                p_count += 1
                cv2.circle(preview, center, 8, (0, 255, 0), -1) 
            elif a_area > limit and a_area > p_area:
                a_count += 1
                cv2.circle(preview, center, 8, (0, 255, 255), -1) 
            else:
                cv2.circle(preview, center, 2, (100, 100, 100), -1)

        final_pines = p_count
        final_aspens = a_count
        
        cv2.putText(preview, f"Pine: {p_count} | Aspen: {a_count}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Tree Tuner", cv2.resize(preview, (1000, 800)))
        if cv2.waitKey(20) == 13: break

    cv2.destroyAllWindows()
    print(f"\nFINAL COUNT - Pine: {final_pines}, Aspen: {final_aspens}")

if __name__ == "__main__":
    if os.path.exists('board.jpg'):
        process_aspens('board.jpg')
    else:
        print("board.jpg not found")