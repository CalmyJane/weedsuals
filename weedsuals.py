import cv2
import pygame
import sys
import numpy as np
import math

def initialize_camera(camera_index=0, capture_w=640, capture_h=360):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    # Force a lower capture resolution to speed things up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_h)
    return cap

def get_fullscreen_display(display_index=0):
    pygame.display.quit()
    pygame.display.init()
    try:
        pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=display_index)
    except pygame.error as e:
        print(f"Error initializing display {display_index}: {e}")
        return None, (0, 0)
    display_info = pygame.display.Info()
    size = (display_info.current_w, display_info.current_h)
    screen = pygame.display.set_mode(size, pygame.FULLSCREEN)
    return screen, size

def apply_filter(frame, filter_type, time):
    if filter_type == 1:  # Invert
        return cv2.bitwise_not(frame)
    elif filter_type == 2:  # Heatmap
        return cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    elif filter_type == 3:  # Fractal overlay
        rows, cols = frame.shape[:2]
        scale = 10
        x = np.linspace(0, cols / scale, cols, dtype=np.float32)
        y = np.linspace(0, rows / scale, rows, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x, y)
        fractal = np.sin(2*np.pi*(x_grid + time/30)) * np.cos(2*np.pi*(y_grid - time/30))
        mn, mx = fractal.min(), fractal.max()
        fractal = ((fractal - mn)/(mx - mn+1e-7))*255
        overlay = np.dstack([fractal]*3).astype(np.uint8)
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    elif filter_type == 4:  # Wave overlay
        rows, cols = frame.shape[:2]
        y_indices, x_indices = np.indices((rows, cols))
        wave = 50*np.sin((x_indices + y_indices + time*10)/100).astype(np.float32)
        overlay = frame.astype(np.float32)
        for c in range(3):
            overlay[:,:,c] += wave
        np.clip(overlay, 0, 255, out=overlay)
        return overlay.astype(np.uint8)
    return frame

def apply_distortion(frame, distortion_type, time, speed, precomputed):
    (x_grid, y_grid, base_r, base_theta, h, w) = precomputed

    # Move 'center' to create a dynamic effect
    center_x = int((w/2) + (w/4)*np.sin(time*speed/15))
    center_y = int((h/2) + (h/4)*np.cos(time*speed/15))

    if distortion_type == 1:  # Wave
        wave = 10*np.sin((y_grid/80)+time/2)*np.sin((x_grid/80)+time/4)
        x_dist = (x_grid + wave).astype(np.float32)
        y_dist = y_grid.astype(np.float32)
        return cv2.remap(frame, x_dist, y_dist, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    elif distortion_type == 2:  # Barrel
        r_dist = base_r + 0.05*np.sin(2*math.pi*(base_r+time/40))
        x_dist = ((r_dist*np.cos(base_theta)+1)*(w-1)/2 - (w/2 - center_x)).astype(np.float32)
        y_dist = ((r_dist*np.sin(base_theta)+1)*(h-1)/2 - (h/2 - center_y)).astype(np.float32)
        return cv2.remap(frame, x_dist, y_dist, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    elif distortion_type == 3:  # Swirl
        swirl_amt = np.sin(time/20)
        theta = base_theta + swirl_amt*np.exp(-base_r**2*2)
        x_dist = ((base_r*np.cos(theta)+1)*(w-1)/2 - (w/2 - center_x)).astype(np.float32)
        y_dist = ((base_r*np.sin(theta)+1)*(h-1)/2 - (h/2 - center_y)).astype(np.float32)
        return cv2.remap(frame, x_dist, y_dist, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    elif distortion_type == 4:  # Ripple
        ripple = 3*np.sin(2*math.pi*(x_grid/100 + time/60))
        x_dist = (x_grid + ripple + center_x/60).astype(np.float32)
        y_dist = (y_grid + center_y/60).astype(np.float32)
        return cv2.remap(frame, x_dist, y_dist, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return frame

def mandala_mirror(frame):
    # Top half (original + horizontal flip)
    top = np.concatenate([frame, cv2.flip(frame, 1)], axis=1)
    # Bottom half (vertical flip of original + vertical+horizontal flip)
    bottom = np.concatenate([cv2.flip(frame, 0), cv2.flip(frame, -1)], axis=1)
    # Full 2×2
    return np.concatenate([top, bottom], axis=0)

def process_frame(frame, flip_x, flip_y, filter_type, distortion_type, time_f, speed, precomputed):
    if flip_x:
        frame = cv2.flip(frame, 1)
    if flip_y:
        frame = cv2.flip(frame, 0)
    # Apply filter + distortion at quarter size
    frame = apply_filter(frame, filter_type, time_f)
    frame = apply_distortion(frame, distortion_type, time_f, speed, precomputed)
    # Then create mandala to effectively double the width and height
    return mandala_mirror(frame)

def find_next_camera(start_index, width=640, height=360):
    index = start_index
    while index <= 10:
        cap = initialize_camera(index, width, height)
        if cap:
            return cap, index
        index += 1
    return None, -1

def main():
    pygame.display.init()
    num_displays = pygame.display.get_num_displays()
    if num_displays <= 0:
        print("No displays detected.")
        sys.exit()

    # Desired capture size is half (or quarter) of final.
    capture_w, capture_h = 640, 360  # e.g., final will be 1280×720 after mirroring

    current_camera = 0
    cap, current_camera = find_next_camera(current_camera, capture_w, capture_h)

    current_display = 0
    screen, (screen_w, screen_h) = get_fullscreen_display(current_display)

    flip_x = False
    flip_y = False
    filter_type = 0
    distortion_type = 0
    time_f = 0.0
    speed = 1

    if cap:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        w, h = capture_w, capture_h

    # Precompute meshgrids used for distortions
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    x_grid = x_grid.astype(np.float32)
    y_grid = y_grid.astype(np.float32)

    x_norm = x_grid / (w/2) - 1
    y_norm = y_grid / (h/2) - 1
    base_r = np.sqrt(x_norm**2 + y_norm**2)
    base_theta = np.arctan2(y_norm, x_norm)

    precomputed = (x_grid, y_grid, base_r, base_theta, h, w)

    clock = pygame.time.Clock()

    while True:
        clock.tick(60)  # aim for up to 60 FPS
        time_f += 0.1
        screen.fill((0,0,0))

        if cap:
            ret, frame = cap.read()
            if not ret:
                print("No frame. Switching camera.")
                cap.release()
                cap, current_camera = find_next_camera(current_camera+1, capture_w, capture_h)
                continue

            # Distort and filter at quarter size
            mandala = process_frame(frame, flip_x, flip_y,
                                    filter_type, distortion_type,
                                    time_f, speed, precomputed)

            # The resulting mandala is now (2×w)×(2×h). 
            # If that matches your final display size, you can just show it.
            # Otherwise, resize to screen_w×screen_h if needed.
            final = cv2.resize(mandala, (screen_w, screen_h), interpolation=cv2.INTER_LINEAR)

            # Convert to RGB for pygame
            final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
            pygame_surf = pygame.surfarray.make_surface(final.swapaxes(0,1))
            screen.blit(pygame_surf, (0,0))

        else:
            # Try next camera or show error
            cap, current_camera = find_next_camera(0, capture_w, capture_h)
            if not cap:
                font = pygame.font.Font(None, 74)
                txt = font.render("No camera found", True, (255, 0, 0))
                rect = txt.get_rect(center=(screen_w//2, screen_h//2))
                screen.blit(txt, rect)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                if cap: cap.release()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:
                    flip_x = not flip_x
                elif event.key == pygame.K_y:
                    flip_y = not flip_y
                elif event.key == pygame.K_c:
                    if cap: cap.release()
                    cap, current_camera = find_next_camera(current_camera+1, capture_w, capture_h)
                elif event.key == pygame.K_v:
                    current_display = (current_display + 1) % num_displays
                    screen, (screen_w, screen_h) = get_fullscreen_display(current_display)
                elif event.key == pygame.K_f:
                    filter_type = (filter_type + 1) % 5
                elif event.key == pygame.K_d:
                    distortion_type = (distortion_type + 1) % 5
                elif event.key == pygame.K_s:
                    speed = (speed % 4) + 1

if __name__ == "__main__":
    main()
