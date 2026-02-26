import cv2
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================

GRID_COLS = 17
GRID_ROWS = 15

# ==========================================
# LOGIC
# ==========================================

perimeter_pixel_points = [] 

def mouse_handler_trace(event, x, y, flags, param):
    global perimeter_pixel_points
    if event == cv2.EVENT_LBUTTONDOWN:
        perimeter_pixel_points.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN: # Undo
        if len(perimeter_pixel_points) > 0: perimeter_pixel_points.pop()

def infer_initial_grid_coords(points):
    if len(points) < 3: return []
    pts = np.array(points, dtype=np.float32)
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    width = max_x - min_x + 1 
    height = max_y - min_y + 1
    
    grid_coords_mutable = []
    for pt in pts:
        norm_x = (pt[0] - min_x) / width
        norm_y = (pt[1] - min_y) / height
        row = int(max(0, min(GRID_ROWS - 1, round(norm_y * (GRID_ROWS - 1)))))
        col = int(max(0, min(GRID_COLS - 1, round(norm_x * (GRID_COLS - 1)))))
        grid_coords_mutable.append([row, col])
        
    return grid_coords_mutable

def get_hex_draw_pos(r, c, width, height):
    # Calculate screen X/Y for a hex grid in the editor window
    # Margins
    margin_x = 60
    margin_y = 60
    usable_w = width - (2 * margin_x)
    usable_h = height - (2 * margin_y)
    
    # Grid math
    # Rows are simple vertical steps
    y_step = usable_h / (GRID_ROWS - 1)
    
    # Cols depend on row parity
    # In pointy top hexes, odd rows are offset by 0.5 widths
    # Max width is (GRID_COLS - 1) + 0.5 (for the offset shift)
    x_step = usable_w / (GRID_COLS - 0.5)
    
    sy = margin_y + (r * y_step)
    sx = margin_x + (c * x_step)
    
    if r % 2 == 1:
        sx += (x_step * 0.5)
        
    return int(sx), int(sy)

def draw_hex_schematic(coords, selected_idx):
    h, w = 700, 1000
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 1. Draw Ghost Grid (The "Slots")
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            sx, sy = get_hex_draw_pos(r, c, w, h)
            # Draw tiny hollow circles for empty slots
            cv2.circle(canvas, (sx, sy), 2, (40, 40, 40), -1)

    # 2. Draw Connecting Lines
    num_pts = len(coords)
    if num_pts > 1:
        points_to_draw = []
        for (r, c) in coords:
            points_to_draw.append(get_hex_draw_pos(r, c, w, h))
        
        pts_array = np.array(points_to_draw, np.int32)
        cv2.polylines(canvas, [pts_array], True, (100, 100, 100), 1)

    # 3. Draw Nodes
    for i, (r, c) in enumerate(coords):
        sx, sy = get_hex_draw_pos(r, c, w, h)
        
        if i == selected_idx:
            # Active: Big Cyan Target
            cv2.circle(canvas, (sx, sy), 18, (255, 255, 0), 2)
            cv2.circle(canvas, (sx, sy), 10, (255, 255, 255), -1)
            
            # Draw Coordinates Label
            label = f"Pt {i}: ({r}, {c})"
            cv2.putText(canvas, label, (20, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Also draw label near point
            cv2.putText(canvas, str(i), (sx+15, sy-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            # Inactive: Small Red
            cv2.circle(canvas, (sx, sy), 6, (0, 0, 255), -1)
            cv2.putText(canvas, str(i), (sx+10, sy-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 200), 1)

    # Instructions
    cv2.putText(canvas, "HEX GRID EDITOR", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, "TAB/BT: Select | WASD: Move | ENTER: Done", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    return canvas

def process_aspens(image_path):
    global perimeter_pixel_points
    
    img = cv2.imread(image_path)
    if img is None: return

    # Display Setup
    view_scale = 1000 / img.shape[1]
    dim = (1000, int(img.shape[0] * view_scale))
    img_display = cv2.resize(img, dim)
    
    # --- STEP 1: TRACE ---
    print("\n" + "="*40)
    print(" STEP 1: TRACE PERIMETER ")
    print("="*40)
    
    cv2.namedWindow("Digitizer")
    cv2.setMouseCallback("Digitizer", mouse_handler_trace)
    
    while True:
        temp = img_display.copy()
        if len(perimeter_pixel_points) > 0:
            pts = np.array(perimeter_pixel_points, np.int32)
            cv2.polylines(temp, [pts], False, (0, 255, 255), 2)
            for i, pt in enumerate(perimeter_pixel_points):
                cv2.circle(temp, tuple(pt), 4, (0, 255, 255), -1)
                # Number the points so user can match them in editor
                if i % 2 == 0 or i==0 or i==len(perimeter_pixel_points)-1:
                     cv2.putText(temp, str(i), (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            if len(perimeter_pixel_points) > 2:
                cv2.line(temp, tuple(perimeter_pixel_points[-1]), tuple(perimeter_pixel_points[0]), (0, 100, 255), 1)

        cv2.putText(temp, f"Points: {len(perimeter_pixel_points)} | ENTER to Edit", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Digitizer", temp)
        
        key = cv2.waitKey(20)
        if key == 13 and len(perimeter_pixel_points) > 5: break 
        elif key == 27: return

    # --- STEP 2: EDIT GRID COORDS ---
    editable_coords = infer_initial_grid_coords(perimeter_pixel_points)
    selected_idx = 0
    
    print("\n" + "="*40)
    print(" STEP 2: HEX GRID EDITOR ")
    print("="*40)
    print(" TAB      : Next Point")
    print(" ` (Tick) : Previous Point (or [ / ] )")
    print(" WASD     : Move Point")
    
    # Keep reference view
    cv2.setMouseCallback("Digitizer", lambda *args: None) 

    while True:
        schematic = draw_hex_schematic(editable_coords, selected_idx)
        cv2.imshow("Hex Grid Editor", schematic)
        
        key = cv2.waitKey(0)
        current_pt = editable_coords[selected_idx]

        if key == 13: break # Enter
        elif key == 27: return
        
        # NAVIGATION
        elif key == 9: # TAB -> Next
            selected_idx = (selected_idx + 1) % len(editable_coords)
        elif key == 96 or key == ord('['): # Backtick (`) or [ -> Prev
            selected_idx = (selected_idx - 1) % len(editable_coords)
        elif key == ord(']'): # ] -> Next
            selected_idx = (selected_idx + 1) % len(editable_coords)
        
        # MOVEMENT
        elif key == ord('w'): current_pt[0] = max(0, current_pt[0] - 1)
        elif key == ord('s'): current_pt[0] = min(GRID_ROWS-1, current_pt[0] + 1)
        elif key == ord('a'): current_pt[1] = max(0, current_pt[1] - 1)
        elif key == ord('d'): current_pt[1] = min(GRID_COLS-1, current_pt[1] + 1)

    cv2.destroyAllWindows()

    # --- FINAL OUTPUT ---
    final_tuple_list = [tuple(pt) for pt in editable_coords]
    
    print("\n" + "="*40)
    print("--- FINAL DIGITIZED SHAPE ---")
    print("Copy this list:")
    print("-" * 20)
    print(f"BOARD_PERIMETER = {final_tuple_list}")
    print("-" * 20)
    print("="*40)

if __name__ == "__main__":
    if os.path.exists('board.jpg'):
        process_aspens('board.jpg')
    else:
        print("Please ensure 'board.jpg' is in the folder.")