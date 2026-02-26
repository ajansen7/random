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
review_click = None

def mouse_handler(event, x, y, flags, param):
    global input_points
    if len(input_points) < 11 and event == cv2.EVENT_LBUTTONDOWN:
        input_points.append([x, y])

def mouse_handler_review(event, x, y, flags, param):
    global review_click
    if event == cv2.EVENT_LBUTTONDOWN:
        review_click = (x, y)

def get_hex_center(row, col, start_x, start_y, radius, off_x=0, off_y=0):
    step_x = radius * np.sqrt(3); step_y = radius * 1.5
    current_x = start_x + (col * step_x) + off_x
    current_y = start_y + (row * step_y) + off_y
    if row % 2 == 1: current_x += (step_x / 2)
    return int(current_x), int(current_y)

def find_forests(grid_matrix):
    forests = {'Pine': [], 'Aspen': []}
    visited = set()
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
    global input_points, review_click
    
    img = cv2.imread(image_path)
    if img is None: 
        print(f"Error: {image_path} not found.")
        return

    saved_profile = {'pine': [], 'aspen': [], 'empty': []}
    
    if os.path.exists(PROFILE_FILENAME):
        print(f"\nLoading training data: {PROFILE_FILENAME}")
        with open(PROFILE_FILENAME, 'r') as f:
            data = json.load(f)
            saved_profile['pine'] = data.get('pine', [])
            saved_profile['aspen'] = data.get('aspen', [])
            saved_profile['empty'] = data.get('empty', [])
            
    if not saved_profile['pine']: saved_profile['pine'].append([0,0,0])
    if not saved_profile['aspen']: saved_profile['aspen'].append([0,0,0])
    if not saved_profile['empty']: saved_profile['empty'].append([20,128,128])

    view_scale = 1000 / img.shape[1]
    dim = (1000, int(img.shape[0] * view_scale))
    img_display = cv2.resize(img, dim)
    
    print("\nSTEP 1: CLICK 11 ANCHORS")
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

    # --- ALIGNMENT ---
    tps = TPSWarp()
    h, w = img.shape[:2]
    
    warp_src = np.array(input_points, dtype=np.float32) * (1/view_scale)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    warp_src_all = np.vstack((warp_src, corners)) 

    print("\nSTEP 2: ALIGN GRID")
    print("TAB: Select Anchor | WASD: Move Anchor | ARROWS: Slide Grid | ENTER: Classify")
    
    dst_height = 1000; MARGIN = 120; aspect_modifier = 1.30 
    global_offset_x = 0; global_offset_y = 0
    base_hex_w = np.sqrt(3); base_hex_h = 1.5
    grid_w_units = (GRID_COLS * base_hex_w) + (0.5 * base_hex_w)
    grid_h_units = (GRID_ROWS * base_hex_h) + 0.5
    base_aspect = grid_w_units / grid_h_units
    
    selected_idx = 10 

    valid_tiles_map = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    valid_tiles_list = [] 
    poly_mask_small = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8)
    poly_pts = np.array([(c, r) for (r, c) in BOARD_SHAPE_COORDS], dtype=np.int32)
    cv2.fillPoly(poly_mask_small, [poly_pts], 1)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if poly_mask_small[r, c] == 1: 
                valid_tiles_map[r, c] = True
                valid_tiles_list.append((r,c))

    bulge_factor = 0.0; bulge_offset_x = 0; bulge_offset_y = 0
    IDX_GHOST_TL = (4, 5); IDX_GHOST_TR = (4, 11)
    IDX_GHOST_BR = (10, 11); IDX_GHOST_BL = (10, 5)

    while True:
        eff_width = int(dst_height * base_aspect * aspect_modifier)
        full_w = eff_width + (MARGIN * 2); full_h = dst_height + (MARGIN * 2)
        hex_r = dst_height / grid_h_units
        sx = MARGIN + (hex_r * np.sqrt(3) / 2); sy = MARGIN + hex_r
        
        dst_anchors = []
        for (r, c) in ANCHORS:
            dx, dy = get_hex_center(r, c, sx, sy, hex_r)
            dst_anchors.append([dx, dy])
            
        ghost_dst = []
        for (r, c) in [IDX_GHOST_TL, IDX_GHOST_TR, IDX_GHOST_BR, IDX_GHOST_BL]:
            dx, dy = get_hex_center(r, c, sx, sy, hex_r)
            ghost_dst.append([dx, dy])
            
        dst_np = np.array(dst_anchors, dtype=np.float32)
        src_np = warp_src.astype(np.float32)
        H_approx, _ = cv2.findHomography(dst_np, src_np)
        ghost_dst_np = np.array([ghost_dst], dtype=np.float32)
        ghost_src_linear = cv2.perspectiveTransform(ghost_dst_np, H_approx)[0]
        
        center_src_base = warp_src[10]
        center_src_adj = center_src_base + np.array([bulge_offset_x, bulge_offset_y])
        
        ghost_src_final = []
        for i in range(4):
            vec = ghost_src_linear[i] - center_src_adj
            new_vec = vec * (1.0 + bulge_factor) 
            ghost_src_final.append(center_src_adj + new_vec)
            
        dst_corners = np.array([[0, 0], [full_w, 0], [full_w, full_h], [0, full_h]], dtype=np.float32)
        all_src = np.vstack((warp_src, np.array(ghost_src_final, dtype=np.float32), corners))
        all_dst = np.vstack((np.array(dst_anchors), np.array(ghost_dst), dst_corners))

        map_x, map_y = tps.solve_and_warp(all_src, all_dst, (full_h, full_w), fast=True)
        warped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

        disp = warped_img.copy()
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if valid_tiles_map[r, c]:
                    cx, cy = get_hex_center(r, c, sx, sy, hex_r, global_offset_x, global_offset_y)
                    if 0 <= cx < full_w and 0 <= cy < full_h:
                        cv2.circle(disp, (cx, cy), int(hex_r*0.4), (0, 255, 255), 1)
                        if (r,c) in ANCHORS:
                            cv2.circle(disp, (cx, cy), 4, (0,0,255), -1)

        status = f"Bulge: {bulge_factor:.2f}"
        cv2.putText(disp, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Main", cv2.resize(disp, (800, 600)))
        
        key = cv2.waitKey(20)
        speed = 4.0; grid_speed = 2
        
        if key == 13: break 
        elif key == 27: return 
        elif key == 9: selected_idx = (selected_idx + 1) % 11
        elif key == ord('w'): warp_src[selected_idx][1] -= speed
        elif key == ord('s'): warp_src[selected_idx][1] += speed
        elif key == ord('a'): warp_src[selected_idx][0] -= speed
        elif key == ord('d'): warp_src[selected_idx][0] += speed
        elif key == ord('v'): bulge_factor -= 0.02
        elif key == ord('b'): bulge_factor += 0.02
        elif key == ord('i'): bulge_offset_y -= speed * 2
        elif key == ord('k'): bulge_offset_y += speed * 2
        elif key == ord('j'): bulge_offset_x -= speed * 2
        elif key == ord('l'): bulge_offset_x += speed * 2
        elif key == 0 or key == 63232: global_offset_y -= grid_speed
        elif key == 1 or key == 63233: global_offset_y += grid_speed
        elif key == 2 or key == 63234: global_offset_x -= grid_speed
        elif key == 3 or key == 63235: global_offset_x += grid_speed

    cv2.destroyAllWindows()

    # --- PHASE 3: CLASSIFICATION & REVIEW ---
    print("\nPreparing High Quality Image...")
    map_x, map_y = tps.make_map((full_h, full_w), quality_scale=0.5)
    warped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    shifted = cv2.pyrMeanShiftFiltering(warped_img, sp=15, sr=30)
    lab_img = cv2.cvtColor(shifted, cv2.COLOR_BGR2Lab)
    
    grid_matrix = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    tile_locations = {} # (r,c) -> (cx, cy, radius)

    print("\n--- REVIEW MODE ---")
    print("Click a tile to toggle: Pine -> Aspen -> Empty")
    print("[S] Save Corrections to Training File")
    print("[ENTER] Finish and Score")
    
    cv2.namedWindow("Main")
    cv2.setMouseCallback("Main", mouse_handler_review)

    first_run = True

    while True:
        # 1. RUN CLASSIFIER with current profile
        pine_arr = np.array(saved_profile['pine'])
        aspen_arr = np.array(saved_profile['aspen'])
        empty_arr = np.array(saved_profile['empty'])
        
        disp_review = warped_img.copy()
        
        for r, c in valid_tiles_list:
            cx, cy = get_hex_center(r, c, sx, sy, hex_r, global_offset_x, global_offset_y)
            tile_locations[(r,c)] = (cx, cy, int(hex_r))
            
            if first_run:
                search_r = int(hex_r * SEARCH_RADIUS_SCALE)
                mask_cell = np.zeros((full_h, full_w), dtype=np.uint8)
                cv2.circle(mask_cell, (cx, cy), search_r, 255, -1)
                mean_lab = cv2.mean(lab_img, mask=mask_cell)[:3]
                
                # KNN
                d_pine = np.min(np.linalg.norm(pine_arr - mean_lab, axis=1))
                d_aspen = np.min(np.linalg.norm(aspen_arr - mean_lab, axis=1))
                d_empty = np.min(np.linalg.norm(empty_arr - mean_lab, axis=1))
                
                closest = min(d_pine, d_aspen, d_empty)
                if closest == d_pine: grid_matrix[r][c] = 1
                elif closest == d_aspen: grid_matrix[r][c] = 2
                else: grid_matrix[r][c] = 0

            # Draw current state
            color = (50, 50, 50) # Empty (Grey)
            label = ""
            if grid_matrix[r][c] == 1: 
                color = (0, 255, 0); label = "P"
            elif grid_matrix[r][c] == 2: 
                color = (0, 255, 255); label = "A"
                
            cv2.circle(disp_review, (cx, cy), int(hex_r*0.5), color, 2)
            cv2.putText(disp_review, label, (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        first_run = False

        # Handle Click
        if review_click is not None:
            # Scale mouse click BACK to image coords
            # Disp size: 1000x800. Image size: full_w x full_h
            scale_x = full_w / 1000.0
            scale_y = full_h / 800.0
            rx = int(review_click[0] * scale_x)
            ry = int(review_click[1] * scale_y)
            
            # Find clicked tile
            for (r,c), (cx, cy, rad) in tile_locations.items():
                if np.linalg.norm([rx-cx, ry-cy]) < rad:
                    new_val = (grid_matrix[r][c] + 1) % 3
                    grid_matrix[r][c] = new_val
                    
                    # EXTRACT AND LEARN
                    search_r = int(hex_r * SEARCH_RADIUS_SCALE)
                    mask_cell = np.zeros((full_h, full_w), dtype=np.uint8)
                    cv2.circle(mask_cell, (cx, cy), search_r, 255, -1)
                    mean_lab = cv2.mean(lab_img, mask=mask_cell)[:3]
                    mean_list = list(mean_lab)
                    
                    if new_val == 1: 
                        saved_profile['pine'].append(mean_list)
                        print("Learned Pine!")
                    elif new_val == 2: 
                        saved_profile['aspen'].append(mean_list)
                        print("Learned Aspen!")
                    elif new_val == 0: 
                        saved_profile['empty'].append(mean_list)
                        print("Learned Empty!")
                    
            review_click = None

        cv2.imshow("Main", cv2.resize(disp_review, (1000, 800)))
        
        key = cv2.waitKey(20)
        if key == 13: break 
        elif key == ord('s'): 
            with open(PROFILE_FILENAME, 'w') as f:
                json.dump(saved_profile, f)
            print(f"Training Saved to {PROFILE_FILENAME}")

    cv2.destroyAllWindows()

    forests = find_forests(grid_matrix)
    p_tot = sum(len(g) for g in forests['Pine'])
    a_tot = sum(len(g) for g in forests['Aspen'])
    print(f"\nFINAL: Pine={p_tot} | Aspen={a_tot}")

if __name__ == "__main__":
    if os.path.exists('board.jpg'):
        process_aspens('board.jpg')
    else:
        print("Please ensure 'board.jpg' is in the folder.")