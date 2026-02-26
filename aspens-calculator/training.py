import cv2
import numpy as np
import os
import json

# ==========================================
# 1. CONFIGURATION
# ==========================================

GRID_COLS = 17 
GRID_ROWS = 15
PROFILE_FILENAME = "aspens_profile.json"

# Verified Perimeter
BOARD_SHAPE_COORDS = [
    (7, 0), (6, 1), (5, 1), (4, 2), (4, 3), (4, 4), (4, 5), (3, 5), (2, 6), (1, 6), 
    (0, 7), (0, 8), (0, 9), (0, 10), (1, 10), (2, 11), (3, 11), (4, 12), (4, 13), (4, 14), (4, 15), 
    (5, 15), (6, 16), (7, 16), (8, 16), (9, 15), (10, 15), (10, 14), (10, 13), (10, 12), 
    (11, 11), (12, 11), (13, 10), (14, 10), (14, 9), (14, 8), (14, 7), (13, 6), (12, 6), 
    (11, 5), (10, 5), (10, 4), (10, 3), (10, 2), (9, 1), (8, 1)
]

# 11 Anchors
IDX_TL = (0, 7); IDX_TR = (0, 10)
IDX_RT = (4, 15); IDX_RM = (7, 16); IDX_RB = (10, 15)
IDX_BR = (14, 10); IDX_BL = (14, 7)
IDX_LB = (10, 2); IDX_LM = (7, 0); IDX_LT = (4, 2)
IDX_CENTER = (7, 8)

ANCHORS = [IDX_TL, IDX_TR, IDX_RT, IDX_RM, IDX_RB, IDX_BR, IDX_BL, IDX_LB, IDX_LM, IDX_LT, IDX_CENTER]
ANCHOR_NAMES = [
    "1. Top-Left", "2. Top-Right", "3. Right-Top", "4. Right-Tip", "5. Right-Bot",
    "6. Bot-Right", "7. Bot-Left", "8. Left-Bot", "9. Left-Tip", "10. Left-Top", "11. CENTER"
]

SEARCH_RADIUS_SCALE = 0.45 

# ==========================================
# TPS ENGINE
# ==========================================

class TPSWarp:
    def __init__(self):
        self.map_x = None; self.map_y = None
    def _U(self, r): return (r**2) * np.log(r + 1e-8)
    def fit(self, src_pts, dst_pts):
        N = src_pts.shape[0]; K = np.zeros((N, N))
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
        w_k = self.weights[:N]; w_p = self.weights[N:]
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
        self.ctrl_dst = dst_pts; self.fit(src_pts, dst_pts)
        scale = 0.08 if fast else 0.5 
        return self.make_map(image_shape, scale)

# ==========================================
# HELPERS
# ==========================================

input_points = []
def mouse_handler(event, x, y, flags, param):
    global input_points
    if len(input_points) < 11 and event == cv2.EVENT_LBUTTONDOWN:
        input_points.append([x, y])

def get_hex_center(row, col, start_x, start_y, radius, off_x=0, off_y=0):
    step_x = radius * np.sqrt(3); step_y = radius * 1.5
    current_x = start_x + (col * step_x) + off_x
    current_y = start_y + (row * step_y) + off_y
    if row % 2 == 1: current_x += (step_x / 2)
    return int(current_x), int(current_y)

def find_forests(grid_matrix):
    forests = {'Pine': [], 'Aspen': []}
    visited = set()
    # Neighbors for odd-r pointy top
    dirs_even = [(-1,-1),(-1,0),(0,-1),(0,1),(1,-1),(1,0)]
    dirs_odd  = [(-1,0),(-1,1),(0,-1),(0,1),(1,0),(1,1)]
    
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            val = grid_matrix[r][c]
            if val == 0 or (r, c) in visited: continue
            tree_type = 'Pine' if val == 1 else 'Aspen'
            current_group = []
            queue = [(r, c)]; visited.add((r, c))
            while queue:
                curr_r, curr_c = queue.pop(0)
                current_group.append((curr_r, curr_c))
                dirs = dirs_odd if curr_r % 2 == 1 else dirs_even
                for dr, dc in dirs:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS and (nr,nc) not in visited:
                        if grid_matrix[nr][nc] == val:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
            forests[tree_type].append(current_group)
    return forests

# ==========================================
# MAIN
# ==========================================

def process_aspens(image_path):
    global input_points
    
    img = cv2.imread(image_path)
    if img is None: 
        print(f"Error: {image_path} not found.")
        return

    # Check for saved profile
    saved_profile = None
    use_saved = False
    if os.path.exists(PROFILE_FILENAME):
        print(f"\nFound saved training data: {PROFILE_FILENAME}")
        choice = input("Do you want to use it? (y/n): ").strip().lower()
        if choice == 'y':
            with open(PROFILE_FILENAME, 'r') as f:
                saved_profile = json.load(f)
            use_saved = True
            # Convert lists back to arrays
            for key in saved_profile:
                saved_profile[key] = np.array(saved_profile[key])

    view_scale = 1000 / img.shape[1]
    dim = (1000, int(img.shape[0] * view_scale))
    img_display = cv2.resize(img, dim)
    
    # --- PHASE 1: ANCHORS ---
    print("\nSTEP 1: CLICK 11 ANCHORS")
    print("Follow the names in the top-left corner.")
    cv2.namedWindow("Source")
    cv2.setMouseCallback("Source", mouse_handler)
    
    while len(input_points) < 11:
        temp = img_display.copy()
        for i, pt in enumerate(input_points):
            cv2.circle(temp, tuple(pt), 6, (0, 255, 0), -1)
            cv2.putText(temp, str(i+1), (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        if len(input_points) < 11:
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
    print("TAB: Select Anchor | WASD: Move Anchor | ARROWS: Slide Grid | ENTER: Done")
    
    dst_height = 1000
    MARGIN = 120 
    aspect_modifier = 1.30 
    
    global_offset_x = 0; global_offset_y = 0
    base_hex_w = np.sqrt(3); base_hex_h = 1.5
    grid_w_units = (GRID_COLS * base_hex_w) + (0.5 * base_hex_w)
    grid_h_units = (GRID_ROWS * base_hex_h) + 0.5
    base_aspect = grid_w_units / grid_h_units
    
    selected_idx = 10 

    # Build Mask
    valid_tiles_map = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    valid_tiles_list = [] # Store list for iteration later
    poly_mask_small = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8)
    poly_pts = np.array([(c, r) for (r, c) in BOARD_SHAPE_COORDS], dtype=np.int32)
    cv2.fillPoly(poly_mask_small, [poly_pts], 1)
    
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if poly_mask_small[r, c] == 1: 
                valid_tiles_map[r, c] = True
                valid_tiles_list.append((r,c))

    while True:
        eff_width = int(dst_height * base_aspect * aspect_modifier)
        full_w = eff_width + (MARGIN * 2); full_h = dst_height + (MARGIN * 2)
        hex_r = dst_height / grid_h_units
        sx = MARGIN + (hex_r * np.sqrt(3) / 2); sy = MARGIN + hex_r
        
        # Calculate ideal positions
        dst_anchors = []
        for (r, c) in ANCHORS:
            dx, dy = get_hex_center(r, c, sx, sy, hex_r)
            dst_anchors.append([dx, dy])
        
        dst_corners = np.array([[0, 0], [full_w, 0], [full_w, full_h], [0, full_h]], dtype=np.float32)
        warp_dst_all = np.vstack((np.array(dst_anchors), dst_corners))
        
        # Re-stack source points (update moved points)
        warp_src_all = np.vstack((warp_src, corners))

        map_x, map_y = tps.solve_and_warp(warp_src_all, warp_dst_all, (full_h, full_w), fast=True)
        warped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

        disp = warped_img.copy()
        
        # Draw Grid
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if valid_tiles_map[r, c]:
                    cx, cy = get_hex_center(r, c, sx, sy, hex_r, global_offset_x, global_offset_y)
                    if 0 <= cx < full_w and 0 <= cy < full_h:
                        cv2.circle(disp, (cx, cy), int(hex_r*0.4), (0, 255, 255), 1)
                        if (r,c) in ANCHORS:
                            cv2.circle(disp, (cx, cy), 4, (0,0,255), -1)

        cv2.putText(disp, f"Offset: {global_offset_x},{global_offset_y}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Preview", cv2.resize(disp, (800, 600)))
        
        # Source Controls (High Visibility)
        src_disp = img_display.copy()
        for i in range(11):
            pt = warp_src[i] * view_scale
            if i == selected_idx:
                cv2.circle(src_disp, tuple(pt.astype(int)), 24, (255, 255, 0), 4)
                cv2.circle(src_disp, tuple(pt.astype(int)), 10, (0, 0, 0), 2)
                cv2.putText(src_disp, ANCHOR_NAMES[i], (int(pt[0])+30, int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
            else:
                cv2.circle(src_disp, tuple(pt.astype(int)), 6, (0, 0, 255), -1)

        cv2.putText(src_disp, "TAB: Select | WASD: Move Anchor | ENTER: Done", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Source", src_disp)

        key = cv2.waitKey(20)
        speed = 4.0; grid_speed = 2
        
        if key == 13: break 
        elif key == 27: return 
        elif key == 9: selected_idx = (selected_idx + 1) % 11
        # WARP SOURCE POINTS
        elif key == ord('w'): warp_src[selected_idx][1] -= speed
        elif key == ord('s'): warp_src[selected_idx][1] += speed
        elif key == ord('a'): warp_src[selected_idx][0] -= speed
        elif key == ord('d'): warp_src[selected_idx][0] += speed
        # GRID SLIDE
        elif key == 0 or key == 63232: global_offset_y -= grid_speed
        elif key == 1 or key == 63233: global_offset_y += grid_speed
        elif key == 2 or key == 63234: global_offset_x -= grid_speed
        elif key == 3 or key == 63235: global_offset_x += grid_speed

    cv2.destroyAllWindows()

    # --- PHASE 3: TRAINING OR CLASSIFICATION ---
    
    # 1. High Quality Prep
    print("\nPreparing Image (Cartoon Filter)...")
    map_x, map_y = tps.make_map((full_h, full_w), quality_scale=0.5)
    warped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    shifted = cv2.pyrMeanShiftFiltering(warped_img, sp=15, sr=30)
    lab_img = cv2.cvtColor(shifted, cv2.COLOR_BGR2Lab)
    
    ref_pine = None; ref_aspen = None; ref_empty = None
    manual_matrix = np.zeros((GRID_ROWS, GRID_COLS), dtype=int) # For "Train on Every" mode

    if use_saved:
        ref_pine = saved_profile['pine']
        ref_aspen = saved_profile['aspen']
        ref_empty = saved_profile['empty']
        print("Profile Loaded.")
    else:
        # TRAIN ON EVERY TILE MODE
        print("\n--- TRAINING MODE ---")
        print("We will step through every tile.")
        print("Press: [1] Pine, [2] Aspen, [3] Empty/Table")
        print("(Press ESC to abort)")
        
        samples_pine = []; samples_aspen = []; samples_empty = []
        
        cv2.namedWindow("Training")
        
        # Sort valid tiles top-to-bottom for natural reading order
        valid_tiles_list.sort()
        
        for idx, (r, c) in enumerate(valid_tiles_list):
            cx, cy = get_hex_center(r, c, sx, sy, hex_r, global_offset_x, global_offset_y)
            search_r = int(hex_r * SEARCH_RADIUS_SCALE)
            
            # Extract Data
            mask_cell = np.zeros((full_h, full_w), dtype=np.uint8)
            cv2.circle(mask_cell, (cx, cy), search_r, 255, -1)
            mean_lab = cv2.mean(lab_img, mask=mask_cell)[:3]
            
            # Interactive Loop
            while True:
                disp_train = warped_img.copy()
                # Highlight current
                cv2.circle(disp_train, (cx, cy), search_r + 5, (0, 0, 255), 4) # Red Ring
                # Dim others
                cv2.addWeighted(disp_train, 0.5, np.zeros(disp_train.shape, dtype=np.uint8), 0.5, 0, disp_train)
                # Draw "Spotlight"
                roi = warped_img[cy-search_r:cy+search_r, cx-search_r:cx+search_r]
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    disp_train[cy-search_r:cy+search_r, cx-search_r:cx+search_r] = roi
                
                cv2.putText(disp_train, f"Tile {idx+1}/{len(valid_tiles_list)}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(disp_train, "1=Pine | 2=Aspen | 3=Empty", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                cv2.imshow("Training", disp_train)
                key = cv2.waitKey(0)
                
                if key == 27: return
                elif key == ord('1'):
                    samples_pine.append(mean_lab)
                    manual_matrix[r][c] = 1
                    break
                elif key == ord('2'):
                    samples_aspen.append(mean_lab)
                    manual_matrix[r][c] = 2
                    break
                elif key == ord('3'):
                    samples_empty.append(mean_lab)
                    manual_matrix[r][c] = 0
                    break
        
        cv2.destroyAllWindows()
        
        # Calculate Profiles
        if len(samples_pine) == 0: samples_pine.append([0,0,0]) # Prevention
        if len(samples_aspen) == 0: samples_aspen.append([0,0,0])
        if len(samples_empty) == 0: samples_empty.append([0,0,0])
        
        ref_pine = np.mean(samples_pine, axis=0)
        ref_aspen = np.mean(samples_aspen, axis=0)
        ref_empty = np.mean(samples_empty, axis=0)
        
        # EXPORT
        export_data = {
            'pine': ref_pine.tolist(),
            'aspen': ref_aspen.tolist(),
            'empty': ref_empty.tolist()
        }
        with open(PROFILE_FILENAME, 'w') as f:
            json.dump(export_data, f)
        print(f"\nTraining Complete! Profile saved to {PROFILE_FILENAME}")

    # --- CLASSIFICATION (Used if loading, or to verify manual) ---
    # Even if we manually classified, we run the logic to verify consistency
    grid_matrix = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    
    for r, c in valid_tiles_list:
        cx, cy = get_hex_center(r, c, sx, sy, hex_r, global_offset_x, global_offset_y)
        search_r = int(hex_r * SEARCH_RADIUS_SCALE)
        mask_cell = np.zeros((full_h, full_w), dtype=np.uint8)
        cv2.circle(mask_cell, (cx, cy), search_r, 255, -1)
        
        mean_lab = cv2.mean(lab_img, mask=mask_cell)[:3]
        
        d_pine  = np.linalg.norm(mean_lab - ref_pine)
        d_aspen = np.linalg.norm(mean_lab - ref_aspen)
        d_empty = np.linalg.norm(mean_lab - ref_empty)
        
        closest = min(d_pine, d_aspen, d_empty)
        
        if closest == d_pine: grid_matrix[r][c] = 1
        elif closest == d_aspen: grid_matrix[r][c] = 2
        else: grid_matrix[r][c] = 0

    # Grouping
    forests = find_forests(grid_matrix)
    
    # Visualization
    disp_final = warped_img.copy()
    
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
    cv2.imshow("Result", cv2.resize(disp_final, (1000, 800)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if os.path.exists('board.jpg'):
        process_aspens('board.jpg')
    else:
        print("Please ensure 'board.jpg' is in the folder.")