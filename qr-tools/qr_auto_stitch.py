import cv2
import numpy as np
from pyzbar.pyzbar import decode
import sys

# --- CONFIGURATION ---
# Add as many filenames as you have. 
# The script will let you choose between them for each step.
SOURCE_FILES = [
    'PXL_20251223_005656289.jpg',  # Image 1
    'LR_QR.jpg',                    # Image 2
    # 'Another_Image.jpg',          # Image 3 (Optional)
]

# Size of ONE quadrant (Final QR will be double this)
QUAD_SIZE = 350

# --- GLOBALS ---
loaded_images = []             # Stores the actual image data
quadrants_data = []            # Stores {'image': img, 'points': []} for each of the 4 steps
quadrants_bw = [None]*4        # Stores processed B&W blocks
offsets = [[0,0], [0,0], [0,0], [0,0]]       
selected_quad = 0 
threshold_val = 128                          

# --- LOAD IMAGES ---
print("Loading images...")
for fname in SOURCE_FILES:
    img = cv2.imread(fname)
    if img is None:
        print(f"Error: Could not load {fname}. Check filename.")
        sys.exit()
    loaded_images.append(img)
print(f"Loaded {len(loaded_images)} source images.")

# --- HELPER FUNCTIONS ---

def select_image_for_step(step_name):
    """
    Asks the user to select which source image to use for the current step.
    """
    win_name = f"Select Image for {step_name}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 800, 600)
    
    selected_idx = 0
    
    while True:
        # Show the currently selected image
        display = loaded_images[selected_idx].copy()
        
        # Overlay Text Instructions
        text = f"STEP: {step_name}"
        cv2.putText(display, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        
        instr1 = f"Showing Image {selected_idx + 1} of {len(loaded_images)}"
        instr2 = "Press [TAB] to switch image"
        instr3 = "Press [ENTER] to use this image"
        
        cv2.putText(display, instr1, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display, instr2, (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display, instr3, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow(win_name, display)
        key = cv2.waitKey(0) & 0xFF
        
        if key == 9: # Tab key
            selected_idx = (selected_idx + 1) % len(loaded_images)
        elif key == 13: # Enter key
            cv2.destroyWindow(win_name)
            return loaded_images[selected_idx]

def get_points_editor(image, step_name):
    """
    Allows user to Click 4 points, then Edit them with WASD.
    """
    points = []
    edit_mode = False
    selected_pt_idx = 0 
    
    win_name = f"Define Corners: {step_name}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 800, 600)
    
    # Callback for initial clicks
    def click_handler(event, x, y, flags, param):
        nonlocal edit_mode
        if not edit_mode and event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])

    cv2.setMouseCallback(win_name, click_handler)
    
    while True:
        display = image.copy()
        
        # Draw Points & Lines
        if len(points) > 0:
            for i, pt in enumerate(points):
                color = (0, 255, 0)
                if edit_mode and i == selected_pt_idx:
                    color = (0, 0, 255) # Red for active
                    cv2.circle(display, tuple(pt), 8, color, -1)
                else:
                    cv2.circle(display, tuple(pt), 5, color, -1)
                
                if i > 0:
                    cv2.line(display, tuple(points[i-1]), tuple(points[i]), (255, 0, 0), 2)
            
            if len(points) == 4:
                cv2.line(display, tuple(points[3]), tuple(points[0]), (255, 0, 0), 2)

        # Instructions
        if not edit_mode:
            cv2.putText(display, f"{step_name}: Click corner {len(points)+1}/4", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if len(points) == 4:
                 cv2.putText(display, "Press SPACE to Edit", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(display, "EDIT MODE: [1-4] Select | [WASD] Move | [ENTER] Done", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(win_name, display)
        key = cv2.waitKey(20) & 0xFF
        
        # Logic
        if not edit_mode:
            if len(points) == 4 and key == 32: # Space
                edit_mode = True
        else:
            if key == ord('1'): selected_pt_idx = 0
            elif key == ord('2'): selected_pt_idx = 1
            elif key == ord('3'): selected_pt_idx = 2
            elif key == ord('4'): selected_pt_idx = 3
            
            step = 1
            # Shift allows faster movement
            # (cv2 keys can be tricky with modifiers, sticking to base speed for stability)
            if key == ord('w'): points[selected_pt_idx][1] -= step
            elif key == ord('s'): points[selected_pt_idx][1] += step
            elif key == ord('a'): points[selected_pt_idx][0] -= step
            elif key == ord('d'): points[selected_pt_idx][0] += step
            
            if key == 13: # Enter
                break
    
    cv2.destroyWindow(win_name)
    return points

def process_bw(image_color, thresh_value):
    gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)
    return bw

def update_composite():
    final_dim = QUAD_SIZE * 2
    canvas = np.ones((final_dim, final_dim), dtype=np.uint8) * 127
    
    # Base positions for TL, TR, BL, BR
    bases = [(0, 0), (0, QUAD_SIZE), (QUAD_SIZE, 0), (QUAD_SIZE, QUAD_SIZE)]
    
    for i in range(4):
        if quadrants_bw[i] is None: continue
        
        img = quadrants_bw[i]
        base_y, base_x = bases[i]
        off_y, off_x = offsets[i]
        
        y1 = base_y + off_y
        x1 = base_x + off_x
        y2 = y1 + QUAD_SIZE
        x2 = x1 + QUAD_SIZE
        
        if x1 < 0 or y1 < 0 or x2 > final_dim or y2 > final_dim: continue
        canvas[y1:y2, x1:x2] = img

    return canvas

def on_trackbar(val):
    global threshold_val
    threshold_val = val
    # Re-process all available quadrants
    for i in range(4):
        if len(quadrants_data) > i:
            data = quadrants_data[i]
            quadrants_bw[i] = process_bw(data['warped'], threshold_val)

# ================= MAIN FLOW =================

step_names = [
    "1. Top-Left Quadrant",
    "2. Top-Right Quadrant",
    "3. Bottom-Left Quadrant",
    "4. Bottom-Right Quadrant"
]

# --- PHASE 1: SELECTION & DEFINITION ---
for step in step_names:
    # 1. Select Image
    img_source = select_image_for_step(step)
    
    # 2. Define Points
    points = get_points_editor(img_source, step)
    
    # 3. Warp immediately to store
    src = np.float32(points)
    dst = np.float32([[0, 0], [QUAD_SIZE-1, 0], [QUAD_SIZE-1, QUAD_SIZE-1], [0, QUAD_SIZE-1]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_source, M, (QUAD_SIZE, QUAD_SIZE))
    
    # Store everything we need
    quadrants_data.append({
        'warped': warped
    })

# Initial Processing
for i in range(4):
    quadrants_bw[i] = process_bw(quadrants_data[i]['warped'], threshold_val)


# --- PHASE 2: ASSEMBLY & FINE TUNING ---
win_name = "Final Assembly"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 800, 800)
cv2.createTrackbar("Darkness", win_name, 128, 255, on_trackbar)

print("\n" + "="*50)
print("FINAL ASSEMBLY MODE")
print("-----------------------")
print(" [1-4]  : Select Quadrant")
print(" [WASD] : Move Selection")
print(" [SLIDER]: Adjust B&W Threshold")
print(" [ENTER]: Save & Exit")
print("="*50)

while True:
    composite = update_composite()
    preview = cv2.cvtColor(composite, cv2.COLOR_GRAY2BGR)
    
    # Draw selection box
    bases = [(0,0), (0,QUAD_SIZE), (QUAD_SIZE,0), (QUAD_SIZE,QUAD_SIZE)]
    by, bx = bases[selected_quad]
    oy, ox = offsets[selected_quad]
    cv2.rectangle(preview, (bx+ox, by+oy), (bx+ox+QUAD_SIZE, by+oy+QUAD_SIZE), (0,0,255), 4)
    
    # Try Decode
    decoded = decode(composite)
    if decoded:
        cv2.putText(preview, "DECODE SUCCESS!", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
        print(f"URL: {decoded[0].data.decode('utf-8')}")
    
    cv2.imshow(win_name, preview)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 13: # Enter
        cv2.imwrite("final_solution.png", composite)
        print("Saved to final_solution.png")
        break
    
    elif key == ord('1'): selected_quad = 0
    elif key == ord('2'): selected_quad = 1
    elif key == ord('3'): selected_quad = 2
    elif key == ord('4'): selected_quad = 3
    
    elif key == ord('w'): offsets[selected_quad][0] -= 1
    elif key == ord('s'): offsets[selected_quad][0] += 1
    elif key == ord('a'): offsets[selected_quad][1] -= 1
    elif key == ord('d'): offsets[selected_quad][1] += 1

cv2.destroyAllWindows()