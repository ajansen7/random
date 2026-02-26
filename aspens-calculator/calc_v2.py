import cv2
import numpy as np
import os

# ==========================================
# 1. BOARD SHAPE DEFINITION
# ==========================================

GRID_COLS = 18 
GRID_ROWS = 15

# Raw digitized list
RAW_PERIMETER = [
    (0, 6), (0, 7), (0, 8), (0, 9), (1, 9), (2, 10), (3, 10), (4, 11), (4, 12), (4, 13), (4, 14), 
    (5, 14), (6, 15), (7, 15), (8, 15), (9, 14), (10, 14), (10, 13), (10, 12), (10, 11), (11, 10), 
    (12, 10), (13, 9), (14, 9), (14, 8), (14, 7), (14, 6), (13, 5), (12, 5), (11, 4), (10, 4), 
    (10, 3), (10, 2), (10, 1), (9, 0), (8, 0), (7, 0), (6, 0), (5, 0), (4, 1), (4, 2), (4, 3), 
    (4, 4), (3, 4), (2, 5), (1, 5)
]

def apply_corrections(raw_list):
    corrected = []
    for (r, c) in raw_list:
        if c == 0: new_c = 0 
        else: new_c = c + 1
        corrected.append((r, new_c))
    return corrected

BOARD_SHAPE_COORDS = apply_corrections(RAW_PERIMETER)

# ==========================================
# 2. CONFIGURATION
# ==========================================

IDX_TOP    = (0, 9)     
IDX_RIGHT  = (7, 17)    
IDX_BOTTOM = (14, 9)    
IDX_LEFT   = (7, 0)     
IDX_CENTER = (7, 9)     

ANCHORS = [IDX_TOP, IDX_RIGHT, IDX_BOTTOM, IDX_LEFT, IDX_CENTER]
ANCHOR_NAMES = ["1. TOP", "2. RIGHT", "3. BOT", "4. LEFT", "5. CENTER"]

# DEFAULT COLORS (Will be tunable)
LOWER_PINE = np.array([30, 40, 40])
UPPER_PINE = np.array([95, 255, 255])
LOWER_ASPEN = np.array([10, 100, 100])
UPPER_ASPEN = np.array([35, 255, 255])

PIXEL_THRESHOLD = 150 # Lowered slightly to catch smaller trees
SEARCH_RADIUS_SCALE = 0.55
BOARD_DARKNESS_THRESHOLD = 110

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
def mouse_handler(event, x, y, flags, param):
    global input_points
    if len(input_points) < 5 and event == cv2.EVENT_LBUTTONDOWN:
        input_points.append([x, y])

def get_hex_center(row, col, start_x, start_y, radius):
    step_x = radius * np.sqrt(3)
    step_y = radius * 1.5
    current_x = start_x + (col * step_x)
    current_y = start_y + (row * step_y)
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

def nothing(x): pass

# ==========================================
# MAIN
# ==========================================

def process_aspens(image_path):
    global input_points, LOWER_PINE, UPPER_PINE, LOWER_ASPEN, UPPER_ASPEN
    
    img = cv2.imread(image_path)
    if img is None: return

    view_scale = 1000 / img.shape[1]
    dim = (1000, int(img.shape[0] * view_scale))
    img_display = cv2.resize(img, dim)
    
    # --- PHASE 1: ANCHORS ---
    print("\nSTEP 1: CLICK 5 ANCHORS")
    print("Top, Right, Bottom, Left, Center")
    cv2.namedWindow("Source")
    cv2.setMouseCallback("Source", mouse_handler)
    
    while len(input_points) < 5:
        temp = img_display.copy()
        for i, pt in enumerate(input_points):
            cv2.circle(temp, tuple(pt), 6, (0, 255, 0), -1)
        if len(input_points) < 5:
             cv2.putText(temp, f"CLICK: {ANCHOR_NAMES[len(input_points)]}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Source", temp)
        if cv2.waitKey(20) == 27: return

    # --- PHASE 2: ALIGNMENT ---
    tps = TPSWarp()
    h, w = img.shape[:2]
    
    warp_src = np.array(input_points, dtype=np.float32) * (1/view_scale)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    warp_src_all = np.vstack((warp_src, corners))

    print("\nSTEP 2: ALIGN GRID")
    print("TAB: Select | WASD: Move Grid | ENTER: Colors")
    
    dst_height = 1000
    MARGIN = 120 
    aspect_modifier = 1.30 
    
    base_hex_w = np.sqrt(3)
    base_hex_h = 1.5
    grid_w_units = (GRID_COLS * base_hex_w) + (0.5 * base_hex_w)
    grid_h_units = (GRID_ROWS * base_hex_h) + 0.5
    base_aspect = grid_w_units / grid_h_units
    
    selected_idx = 4 

    # Pre-calculate Mask
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
        
        dst_anchors = []
        for (r, c) in ANCHORS:
            dx, dy = get_hex_center(r, c, sx, sy, hex_r)
            dst_anchors.append([dx, dy])
        
        dst_corners = np.array([[0, 0], [full_w, 0], [full_w, full_h], [0, full_h]], dtype=np.float32)
        warp_dst_all = np.vstack((np.array(dst_anchors), dst_corners))
        
        map_x, map_y = tps.solve_and_warp(warp_src_all, warp_dst_all, (full_h, full_w), fast=True)
        warped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

        disp = warped_img.copy()
        
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if valid_tiles_map[r, c]:
                    cx, cy = get_hex_center(r, c, sx, sy, hex_r)
                    if 0 <= cx < full_w and 0 <= cy < full_h:
                        cv2.circle(disp, (cx, cy), int(hex_r*0.4), (0, 255, 255), 1)
                        if (r,c) in ANCHORS:
                            cv2.circle(disp, (cx, cy), 4, (0,0,255), -1)

        cv2.putText(disp, f"Aspect: {aspect_modifier:.2f}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Preview", cv2.resize(disp, (800, 600)))
        
        src_disp = img_display.copy()
        for i in range(5):
            pt = warp_src[i] * view_scale
            if i == selected_idx:
                cv2.circle(src_disp, tuple(pt.astype(int)), 24, (255, 255, 0), 4) 
                cv2.putText(src_disp, ANCHOR_NAMES[i], (int(pt[0])+30, int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
            else:
                cv2.circle(src_disp, tuple(pt.astype(int)), 6, (0, 0, 255), -1)

        cv2.putText(src_disp, "TAB: Select | WASD: Align | ENTER: Done", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Source", src_disp)

        key = cv2.waitKey(20)
        speed = 4.0
        if key == 13: break 
        elif key == 27: return 
        elif key == 9: selected_idx = (selected_idx + 1) % 5
        elif key == ord('w'): warp_src[selected_idx][1] -= speed
        elif key == ord('s'): warp_src[selected_idx][1] += speed
        elif key == ord('a'): warp_src[selected_idx][0] -= speed
        elif key == ord('d'): warp_src[selected_idx][0] += speed
        elif key == ord('z'): aspect_modifier -= 0.02
        elif key == ord('x'): aspect_modifier += 0.02

    cv2.destroyAllWindows()

    # --- PHASE 3: COLOR TUNING ---
    print("\nSTEP 3: TUNE COLORS")
    
    # Calculate high-quality warp once
    map_x, map_y = tps.make_map((full_h, full_w), quality_scale=0.5)
    warped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    hsv_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2HSV)

    cv2.namedWindow("Color Tuner")
    
    # Trackbars for PINE
    cv2.createTrackbar("PINE Hue Min", "Color Tuner", LOWER_PINE[0], 179, nothing)
    cv2.createTrackbar("PINE Hue Max", "Color Tuner", UPPER_PINE[0], 179, nothing)
    cv2.createTrackbar("PINE Sat Min", "Color Tuner", LOWER_PINE[1], 255, nothing)
    
    # Trackbars for ASPEN
    cv2.createTrackbar("ASPEN Hue Min", "Color Tuner", LOWER_ASPEN[0], 179, nothing)
    cv2.createTrackbar("ASPEN Hue Max", "Color Tuner", UPPER_ASPEN[0], 179, nothing)
    cv2.createTrackbar("ASPEN Sat Min", "Color Tuner", LOWER_ASPEN[1], 255, nothing)

    active_view = "PINE" 

    while True:
        # Get Trackbar positions
        p_h_min = cv2.getTrackbarPos("PINE Hue Min", "Color Tuner")
        p_h_max = cv2.getTrackbarPos("PINE Hue Max", "Color Tuner")
        p_s_min = cv2.getTrackbarPos("PINE Sat Min", "Color Tuner")
        
        a_h_min = cv2.getTrackbarPos("ASPEN Hue Min", "Color Tuner")
        a_h_max = cv2.getTrackbarPos("ASPEN Hue Max", "Color Tuner")
        a_s_min = cv2.getTrackbarPos("ASPEN Sat Min", "Color Tuner")

        # Update Globals
        LOWER_PINE = np.array([p_h_min, p_s_min, 40])
        UPPER_PINE = np.array([p_h_max, 255, 255])
        
        LOWER_ASPEN = np.array([a_h_min, a_s_min, 100])
        UPPER_ASPEN = np.array([a_h_max, 255, 255])

        # Generate Masks
        mask_pine = cv2.inRange(hsv_img, LOWER_PINE, UPPER_PINE)
        mask_aspen = cv2.inRange(hsv_img, LOWER_ASPEN, UPPER_ASPEN)

        if active_view == "PINE":
            view = mask_pine
        else:
            view = mask_aspen

        # Overlay instructions
        disp_color = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
        cv2.putText(disp_color, f"VIEWING: {active_view} (Press 1 for Pine, 2 for Aspen)", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(disp_color, "Adjust Sliders until trees are WHITE. ENTER to Finish.", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow("Color Tuner", cv2.resize(disp_color, (800, 600)))
        
        key = cv2.waitKey(20)
        if key == 13: break
        elif key == ord('1'): active_view = "PINE"
        elif key == ord('2'): active_view = "ASPEN"

    cv2.destroyAllWindows()
    
    # --- FINAL CALC ---
    print("Calculating final score...")
    
    # Re-calc masks with final settings
    mask_pine = cv2.inRange(hsv_img, LOWER_PINE, UPPER_PINE)
    mask_aspen = cv2.inRange(hsv_img, LOWER_ASPEN, UPPER_ASPEN)
    
    grid_matrix = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    disp_final = warped_img.copy()

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            # COOKIE CUTTER MASK
            if not valid_tiles_map[row, col]: continue

            cx, cy = get_hex_center(row, col, sx, sy, hex_r)
            if not (0 <= cx < full_w and 0 <= cy < full_h): continue
            
            search_r = int(hex_r * SEARCH_RADIUS_SCALE)
            mask_cell = np.zeros((full_h, full_w), dtype=np.uint8)
            cv2.circle(mask_cell, (cx, cy), search_r, 255, -1)
            
            n_pine = cv2.countNonZero(cv2.bitwise_and(mask_pine, mask_pine, mask=mask_cell))
            n_aspen = cv2.countNonZero(cv2.bitwise_and(mask_aspen, mask_aspen, mask=mask_cell))

            if n_pine > PIXEL_THRESHOLD and n_pine > n_aspen:
                 grid_matrix[row][col] = 1
            elif n_aspen > PIXEL_THRESHOLD and n_aspen > n_pine:
                 grid_matrix[row][col] = 2

    forests = find_forests(grid_matrix)
    
    print("\nFINAL SCORES:")
    p_tot = 0
    for i, g in enumerate(forests['Pine']):
        p_tot += len(g)
        for r, c in g:
            cx, cy = get_hex_center(r, c, sx, sy, hex_r)
            cv2.circle(disp_final, (cx, cy), int(hex_r*0.5), (0, 100, 0), -1)
            cv2.putText(disp_final, f"P{i+1}", (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
    a_tot = 0
    for i, g in enumerate(forests['Aspen']):
        a_tot += len(g)
        for r, c in g:
            cx, cy = get_hex_center(r, c, sx, sy, hex_r)
            cv2.circle(disp_final, (cx, cy), int(hex_r*0.5), (0, 200, 255), -1)
            cv2.putText(disp_final, f"A{i+1}", (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

    print(f"Pine: {p_tot}")
    print(f"Aspen: {a_tot}")
    
    cv2.imshow("Final", cv2.resize(disp_final, (1000, 800)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if os.path.exists('board.jpg'):
        process_aspens('board.jpg')
    else:
        print("Please ensure 'board.jpg' is in the folder.")