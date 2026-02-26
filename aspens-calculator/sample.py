import cv2
import numpy as np
import os

# ==========================================
# 1. THE GOLDEN CONFIGURATION
# ==========================================

GRID_COLS = 17 
GRID_ROWS = 15

# YOUR VERIFIED PERIMETER (Do not change)
BOARD_SHAPE_COORDS = [
    (7, 0), (6, 1), (5, 1), (4, 2), (4, 3), (4, 4), (4, 5), (3, 5), (2, 6), (1, 6), 
    (0, 7), (0, 8), (0, 9), (0, 10), (1, 10), (2, 11), (3, 11), (4, 12), (4, 13), (4, 14), (4, 15), 
    (5, 15), (6, 16), (7, 16), (8, 16), (9, 15), (10, 15), (10, 14), (10, 13), (10, 12), 
    (11, 11), (12, 11), (13, 10), (14, 10), (14, 9), (14, 8), (14, 7), (13, 6), (12, 6), 
    (11, 5), (10, 5), (10, 4), (10, 3), (10, 2), (9, 1), (8, 1)
]

# 11-POINT ANCHOR SYSTEM
# Indices (Row, Col) mapped to the corners of the shape above
IDX_TL      = (0, 7)    # Top-Left Corner
IDX_TR      = (0, 10)   # Top-Right Corner
IDX_RT      = (4, 15)   # Right Arm Top Shoulder
IDX_RM      = (7, 16)   # Right Arm Tip
IDX_RB      = (10, 15)  # Right Arm Bot Shoulder
IDX_BR      = (14, 10)  # Bot-Right Corner
IDX_BL      = (14, 7)   # Bot-Left Corner
IDX_LB      = (10, 2)   # Left Arm Bot Shoulder
IDX_LM      = (7, 0)    # Left Arm Tip
IDX_LT      = (4, 2)    # Left Arm Top Shoulder
IDX_CENTER  = (7, 8)    # Center

ANCHORS = [
    IDX_TL, IDX_TR,
    IDX_RT, IDX_RM, IDX_RB,
    IDX_BR, IDX_BL,
    IDX_LB, IDX_LM, IDX_LT,
    IDX_CENTER
]

ANCHOR_NAMES = [
    "1. Top-Left", "2. Top-Right",
    "3. Right-Top", "4. Right-Tip", "5. Right-Bot",
    "6. Bot-Right", "7. Bot-Left",
    "8. Left-Bot", "9. Left-Tip", "10. Left-Top",
    "11. CENTER"
]

SEARCH_RADIUS_SCALE = 0.45 

# ==========================================
# TPS ENGINE
# ==========================================

class TPSWarp:
    def __init__(self):
        self.map_x = None
        self.map_y = None
    def _U(self, r): return (r**2) * np.log(r + 1e-8)
    def fit(self, src_pts, dst_pts):
        N = src_pts.shape[0]
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                r = np.linalg.norm(dst_pts[i] - dst_pts[j])
                K[i, j] = self._U(r)
        P = np.hstack((np.ones((N, 1)), dst_pts))
        L = np.vstack((np.hstack((K, P)), np.hstack((P.T, np.zeros((3, 3))))))
        V = np.vstack((src_pts, np.zeros((3, 2))))
        L += np.eye(L.shape[0]) * 1e-4
        self.weights = np.linalg.solve(L, V)
    def make_map(self, out_shape, quality_scale=0.1):
        h, w = out_shape[:2]
        small_h, small_w = int(h * quality_scale), int(w * quality_scale)
        grid_y, grid_x = np.mgrid[0:small_h, 0:small_w]
        pts = np.vstack((grid_x.ravel() * (1/quality_scale), grid_y.ravel() * (1/quality_scale))).T
        N = self.weights.shape[0] - 3
        ctrl_pts = self.ctrl_dst
        w_k = self.weights[:N]
        w_p = self.weights[N:]
        affine = np.hstack((np.ones((pts.shape[0], 1)), pts)) @ w_p
        non_linear = np.zeros_like(affine)
        for i in range(N):
            dist = np.linalg.norm(pts - ctrl_pts[i], axis=1)
            u_val = self._U(dist)
            non_linear += u_val[:, np.newaxis] * w_k[i]
        map_coords = affine + non_linear
        map_x_small = map_coords[:, 0].reshape(small_h, small_w).astype(np.float32)
        map_y_small = map_coords[:, 1].reshape(small_h, small_w).astype(np.float32)
        self.map_x = cv2.resize(map_x_small, (w, h), interpolation=cv2.INTER_LINEAR)
        self.map_y = cv2.resize(map_y_small, (w, h), interpolation=cv2.INTER_LINEAR)
        return self.map_x, self.map_y
    def solve_and_warp(self, src_pts, dst_pts, image_shape, fast=True):
        self.ctrl_dst = dst_pts 
        self.fit(src_pts, dst_pts)
        scale = 0.08 if fast else 0.5 
        return self.make_map(image_shape, scale)

# ==========================================
# HELPERS
# ==========================================

input_points = []
# Tracks the current mouse position for the "Hover to Teach" phase
mouse_curr_pos = (0, 0) 

def mouse_handler(event, x, y, flags, param):
    global input_points, mouse_curr_pos
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_curr_pos = (x, y)
    
    # Only capture clicks during Phase 1
    if len(input_points) < 11 and event == cv2.EVENT_LBUTTONDOWN:
        input_points.append([x, y])

def get_hex_center(row, col, start_x, start_y, radius, off_x=0, off_y=0):
    step_x = radius * np.sqrt(3)
    step_y = radius * 1.5
    current_x = start_x + (col * step_x) + off_x
    current_y = start_y + (row * step_y) + off_y
    if row % 2 == 1: current_x += (step_x / 2)
    return int(current_x), int(current_y)

def get_neighbors(r, c):
    neighbors = []
    directions_even = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
    directions_odd  = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
    dirs = directions_odd if r % 2 == 1 else directions_even
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
            neighbors.append((nr, nc))
    return neighbors

def find_forests(grid_matrix):
    forests = {'Pine': [], 'Aspen': []}
    visited = set()
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            val = grid_matrix[r][c]
            if val == 0 or (r, c) in visited: continue
            tree_type = 'Pine' if val == 1 else 'Aspen'
            current_group = []
            queue = [(r, c)]
            visited.add((r, c))
            while queue:
                curr_r, curr_c = queue.pop(0)
                current_group.append((curr_r, curr_c))
                for nr, nc in get_neighbors(curr_r, curr_c):
                    if (nr, nc) not in visited and grid_matrix[nr][nc] == val:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            forests[tree_type].append(current_group)
    forests['Pine'].sort(key=len, reverse=True)
    forests['Aspen'].sort(key=len, reverse=True)
    return forests

# ==========================================
# MAIN
# ==========================================

def process_aspens(image_path):
    global input_points, mouse_curr_pos
    
    img = cv2.imread(image_path)
    if img is None: 
        print(f"Error: {image_path} not found.")
        return

    view_scale = 1000 / img.shape[1]
    dim = (1000, int(img.shape[0] * view_scale))
    img_display = cv2.resize(img, dim)
    
    # --- PHASE 1: CLICK 11 ANCHORS ---
    print("\nSTEP 1: CLICK 11 ANCHORS")
    print("Follow the instruction on the image.")
    cv2.namedWindow("Main")
    cv2.setMouseCallback("Main", mouse_handler)
    
    while len(input_points) < 11:
        temp = img_display.copy()
        for i, pt in enumerate(input_points):
            cv2.circle(temp, tuple(pt), 6, (0, 255, 0), -1)
            cv2.putText(temp, str(i+1), (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        if len(input_points) < 11:
             msg = f"CLICK: {ANCHOR_NAMES[len(input_points)]}"
             cv2.rectangle(temp, (0,0), (600, 60), (0,0,0), -1)
             cv2.putText(temp, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow("Main", temp)
        if cv2.waitKey(20) == 27: return

    # --- PHASE 2: ALIGN GRID ---
    tps = TPSWarp()
    h, w = img.shape[:2]
    
    warp_src = np.array(input_points, dtype=np.float32) * (1/view_scale)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    warp_src_all = np.vstack((warp_src, corners))

    print("\nSTEP 2: ALIGN GRID")
    print("TAB: Select Anchor | WASD: Warp | ARROWS: Slide Grid | ENTER: Teach")
    
    dst_height = 1000
    MARGIN = 120 
    aspect_modifier = 1.30 
    
    global_offset_x = 0
    global_offset_y = 0
    
    base_hex_w = np.sqrt(3)
    base_hex_h = 1.5
    grid_w_units = (GRID_COLS * base_hex_w) + (0.5 * base_hex_w)
    grid_h_units = (GRID_ROWS * base_hex_h) + 0.5
    base_aspect = grid_w_units / grid_h_units
    
    selected_idx = 10 # Start at Center

    # Build Mask from Board Shape
    valid_tiles_map = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    poly_mask_small = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8)
    poly_pts = np.array([(c, r) for (r, c) in BOARD_SHAPE_COORDS], dtype=np.int32)
    cv2.fillPoly(poly_mask_small, [poly_pts], 1)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if poly_mask_small[r, c] == 1: valid_tiles_map[r, c] = True

    while True:
        eff_width = int(dst_height * base_aspect * aspect_modifier)
        full_w = eff_width + (MARGIN * 2)
        full_h = dst_height + (MARGIN * 2)
        
        hex_r = dst_height / grid_h_units
        sx = MARGIN + (hex_r * np.sqrt(3) / 2)
        sy = MARGIN + hex_r
        
        # Calculate ideal positions for 11 anchors
        dst_anchors = []
        for (r, c) in ANCHORS:
            dx, dy = get_hex_center(r, c, sx, sy, hex_r)
            dst_anchors.append([dx, dy])
        
        dst_corners = np.array([[0, 0], [full_w, 0], [full_w, full_h], [0, full_h]], dtype=np.float32)
        warp_dst_all = np.vstack((np.array(dst_anchors), dst_corners))
        
        map_x, map_y = tps.solve_and_warp(warp_src_all, warp_dst_all, (full_h, full_w), fast=True)
        warped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

        disp = warped_img.copy()
        
        # Draw Grid
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if valid_tiles_map[r, c]:
                    cx, cy = get_hex_center(r, c, sx, sy, hex_r, global_offset_x, global_offset_y)
                    if 0 <= cx < full_w and 0 <= cy < full_h:
                        # Draw Hex Outline
                        cv2.circle(disp, (cx, cy), int(hex_r*0.4), (0, 255, 255), 1)
                        if (r,c) in ANCHORS:
                            cv2.circle(disp, (cx, cy), 4, (0,0,255), -1)

        cv2.putText(disp, f"Offset: {global_offset_x},{global_offset_y} | Aspect: {aspect_modifier:.2f}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Main", cv2.resize(disp, (800, 600)))
        
        key = cv2.waitKey(20)
        speed = 4.0
        grid_speed = 2
        
        if key == 13: break 
        elif key == 27: return 
        elif key == 9: selected_idx = (selected_idx + 1) % 11
        elif key == ord('w'): warp_src[selected_idx][1] -= speed
        elif key == ord('s'): warp_src[selected_idx][1] += speed
        elif key == ord('a'): warp_src[selected_idx][0] -= speed
        elif key == ord('d'): warp_src[selected_idx][0] += speed
        elif key == ord('z'): aspect_modifier -= 0.02
        elif key == ord('x'): aspect_modifier += 0.02
        # Arrows
        elif key == 0 or key == 63232: global_offset_y -= grid_speed
        elif key == 1 or key == 63233: global_offset_y += grid_speed
        elif key == 2 or key == 63234: global_offset_x -= grid_speed
        elif key == 3 or key == 63235: global_offset_x += grid_speed

    # --- PHASE 3: TEACH BY EXAMPLE ---
    print("\nSTEP 3: TEACH COLORS")
    print("Applying Smooth Filter (1-3s)...")
    
    map_x, map_y = tps.make_map((full_h, full_w), quality_scale=0.5)
    warped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    shifted = cv2.pyrMeanShiftFiltering(warped_img, sp=15, sr=30)
    # Using LAB color space for better human perceptual matching
    lab_img = cv2.cvtColor(shifted, cv2.COLOR_BGR2Lab)
    
    samples_pine = []
    samples_aspen = []
    samples_empty = []
    
    print("HOVER & PRESS: 1=Pine, 2=Aspen, 3=Empty. ENTER to Finish.")

    while True:
        disp_train = shifted.copy()
        
        # Cursor Mapping
        mx, my = mouse_curr_pos
        disp_h, disp_w = 600, 800
        
        # Scale mouse (Window) -> Image
        scale_x = warped_img.shape[1] / disp_w
        scale_y = warped_img.shape[0] / disp_h
        
        real_x = int(mx * scale_x)
        real_y = int(my * scale_y)
        
        # Clamp
        real_x = max(0, min(warped_img.shape[1]-1, real_x))
        real_y = max(0, min(warped_img.shape[0]-1, real_y))
        
        curr_color = lab_img[real_y, real_x]
        
        # UI
        disp_view = cv2.resize(disp_train, (disp_w, disp_h))
        cv2.circle(disp_view, (mx, my), 10, (255, 255, 255), 2)
        cv2.rectangle(disp_view, (0,0), (400, 120), (0,0,0), -1)
        cv2.putText(disp_view, f"Pines: {len(samples_pine)} samples", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(disp_view, f"Aspens: {len(samples_aspen)} samples", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        cv2.putText(disp_view, f"Empty: {len(samples_empty)} samples", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("Main", disp_view)
        
        key = cv2.waitKey(20)
        if key == 13: 
            if len(samples_pine) == 0 or len(samples_aspen) == 0:
                print("Warning: Sample at least one Pine and one Aspen!")
                continue
            break
        elif key == ord('1'): samples_pine.append(curr_color)
        elif key == ord('2'): samples_aspen.append(curr_color)
        elif key == ord('3'): samples_empty.append(curr_color)

    cv2.destroyAllWindows()
    
    # --- FINAL CLASSIFICATION ---
    print("Classifying...")
    
    ref_pine = np.mean(samples_pine, axis=0)
    ref_aspen = np.mean(samples_aspen, axis=0)
    
    if len(samples_empty) > 0:
        ref_empty = np.mean(samples_empty, axis=0)
    else:
        # Fallback dark color if no empty samples
        ref_empty = np.array([20, 128, 128]) 
    
    grid_matrix = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    disp_final = warped_img.copy()

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            if not valid_tiles_map[row, col]: continue

            cx, cy = get_hex_center(row, col, sx, sy, hex_r, global_offset_x, global_offset_y)
            if not (0 <= cx < full_w and 0 <= cy < full_h): continue
            
            search_r = int(hex_r * SEARCH_RADIUS_SCALE)
            mask_cell = np.zeros((full_h, full_w), dtype=np.uint8)
            cv2.circle(mask_cell, (cx, cy), search_r, 255, -1)
            
            # Mean LAB Color
            mean_lab = cv2.mean(lab_img, mask=mask_cell)[:3]
            
            # Distance
            d_pine  = np.linalg.norm(mean_lab - ref_pine)
            d_aspen = np.linalg.norm(mean_lab - ref_aspen)
            d_empty = np.linalg.norm(mean_lab - ref_empty)
            
            closest = min(d_pine, d_aspen, d_empty)
            
            if closest == d_pine: grid_matrix[row][col] = 1
            elif closest == d_aspen: grid_matrix[row][col] = 2

    forests = find_forests(grid_matrix)
    
    p_tot = sum(len(g) for g in forests['Pine'])
    for g in forests['Pine']:
        for r, c in g:
            cx, cy = get_hex_center(r, c, sx, sy, hex_r, global_offset_x, global_offset_y)
            cv2.circle(disp_final, (cx, cy), int(hex_r*0.5), (0, 100, 0), -1)
            cv2.putText(disp_final, "P", (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
    a_tot = sum(len(g) for g in forests['Aspen'])
    for g in forests['Aspen']:
        for r, c in g:
            cx, cy = get_hex_center(r, c, sx, sy, hex_r, global_offset_x, global_offset_y)
            cv2.circle(disp_final, (cx, cy), int(hex_r*0.5), (0, 200, 255), -1)
            cv2.putText(disp_final, "A", (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    print(f"\nFINAL: Pine={p_tot} | Aspen={a_tot}")
    cv2.imshow("Final", cv2.resize(disp_final, (1000, 800)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if os.path.exists('board.jpg'):
        process_aspens('board.jpg')
    else:
        print("Please ensure 'board.jpg' is in the folder.")