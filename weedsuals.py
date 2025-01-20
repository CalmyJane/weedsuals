import cv2
import pygame
import sys
import numpy as np
import math
import random

def initialize_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)  # Initialize the camera capture
    if not cap.isOpened():
        return None
    return cap

def get_fullscreen_display(display_index=0):
    pygame.display.quit()  # Reset the display system
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
    if filter_type == 1:  # Invert Colors
        return cv2.bitwise_not(frame)
    elif filter_type == 2:  # Heatmap Effect
        return cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    elif filter_type == 3:  # Organic Fractal Noise Overlay
        rows, cols, _ = frame.shape
        scale = 10  # Larger scale for zoomed-in fractals
        x = np.linspace(0, cols / scale, cols)
        y = np.linspace(0, rows / scale, rows)
        x_grid, y_grid = np.meshgrid(x, y)
        fractal_noise = np.sin(2 * np.pi * (x_grid + time / 30)) * np.cos(2 * np.pi * (y_grid - time / 30))
        fractal_noise = (fractal_noise - fractal_noise.min()) / (fractal_noise.max() - fractal_noise.min()) * 255
        overlay = np.dstack([fractal_noise] * 3).astype(np.uint8)
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    elif filter_type == 4:  # Wave Overlay
        rows, cols, _ = frame.shape
        overlay = np.zeros_like(frame, dtype=np.float32)
        y_indices, x_indices = np.indices((rows, cols))
        wave = 50 * np.sin((x_indices + y_indices + time * 10) / 100)
        overlay[:, :, 0] = wave
        overlay[:, :, 1] = wave
        overlay[:, :, 2] = wave
        overlay = np.clip(frame.astype(np.float32) + overlay, 0, 255).astype(np.uint8)
        return overlay
    return frame  # No filter

def apply_distortion(frame, distortion_type, time, speed):
    h, w = frame.shape[:2]
    
    # Adjust center movement for smoother and slower oscillation
    center_x = int((w / 2) + (w / 4) * np.sin(time * speed / 15))
    center_y = int((h / 2) + (h / 4) * np.cos(time * speed / 15))

    if distortion_type == 1:  # Dynamic Wave Distortion
        x = np.arange(w)
        y = np.arange(h)
        x_grid, y_grid = np.meshgrid(x, y)
        wave = 10 * np.sin((y_grid / 80) + time / 2) * np.sin((x_grid / 80) + time / 4)
        x_distorted = (x_grid + wave).astype(np.float32)
        y_distorted = y_grid.astype(np.float32)
        return cv2.remap(frame, x_distorted, y_distorted, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    elif distortion_type == 2:  # Moving Barrel Distortion with Mirroring
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        x_grid, y_grid = np.meshgrid(x, y)
        r = np.sqrt(x_grid**2 + y_grid**2)
        theta = np.arctan2(y_grid, x_grid)
        r_distorted = r + 0.05 * np.sin(2 * math.pi * (r + time / 40))
        x_distorted = ((r_distorted * np.cos(theta) + 1) * (w - 1) / 2 - (w / 2 - center_x)).astype(np.float32)
        y_distorted = ((r_distorted * np.sin(theta) + 1) * (h - 1) / 2 - (h / 2 - center_y)).astype(np.float32)
        return cv2.remap(frame, x_distorted, y_distorted, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    elif distortion_type == 3:  # Swirl Effect
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        x_grid, y_grid = np.meshgrid(x, y)
        r = np.sqrt(x_grid**2 + y_grid**2)
        theta = np.arctan2(y_grid, x_grid) + 1.0 * np.sin(time / 20) * np.exp(-r**2 * 2)
        x_distorted = ((r * np.cos(theta) + 1) * (w - 1) / 2 - (w / 2 - center_x)).astype(np.float32)
        y_distorted = ((r * np.sin(theta) + 1) * (h - 1) / 2 - (h / 2 - center_y)).astype(np.float32)
        return cv2.remap(frame, x_distorted, y_distorted, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    elif distortion_type == 4:  # Ripple Effect
        x = np.arange(w)
        y = np.arange(h)
        x_grid, y_grid = np.meshgrid(x, y)
        ripple = 3 * np.sin(2 * math.pi * (x_grid / 100 + time / 60))
        x_distorted = (x_grid + ripple + center_x / 60).astype(np.float32)
        y_distorted = (y_grid + center_y / 60).astype(np.float32)
        return cv2.remap(frame, x_distorted, y_distorted, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return frame  # No distortion

def process_frame(frame, flip_x=False, flip_y=False, filter_type=0, distortion_type=0, time=0, speed=1):
    # Apply optional flips
    if flip_x:
        frame = cv2.flip(frame, 1)  # Horizontal flip
    if flip_y:
        frame = cv2.flip(frame, 0)  # Vertical flip

    # Apply filters and distortions
    frame = apply_filter(frame, filter_type, time)
    frame = apply_distortion(frame, distortion_type, time, speed)

    # Create the mandala effect by mirroring
    top_half = np.concatenate((frame, cv2.flip(frame, 1)), axis=1)  # Original + Horizontal Flip
    bottom_half = np.concatenate((cv2.flip(frame, 0), cv2.flip(frame, -1)), axis=1)  # Vertical + Both Axes Flip
    mandala = np.concatenate((top_half, bottom_half), axis=0)  # Combine top and bottom halves
    return mandala

def find_next_camera(start_index, excluded_camera=-1):
    index = start_index
    while True:
        cap = initialize_camera(index)
        if cap:
            return cap, index
        index += 1
        if index > 10:  # Arbitrary limit to avoid infinite loop
            return None, -1

def main():
    current_camera = 0
    current_display = 0
    pygame.display.init()
    num_displays = pygame.display.get_num_displays()
    if num_displays <= 0:
        print("No displays detected.")
        sys.exit()

    cap, current_camera = find_next_camera(current_camera)
    screen, size = get_fullscreen_display(current_display)
    flip_x = False  # Initial state for Flip X
    flip_y = False  # Initial state for Flip Y
    filter_type = 0  # No filter initially
    distortion_type = 0  # No distortion initially
    time = 0  # Time for dynamic effects
    speed = 1  # Initial distortion speed

    while True:
        time += 0.1  # Increment time for dynamic effects
        if cap:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Switching to next camera.")
                cap.release()
                cap, current_camera = find_next_camera(current_camera + 1)
                continue

            # Resize the camera feed to fit 1/4th of the screen
            frame = cv2.resize(frame, (size[0] // 2, size[1] // 2))  # Adjust frame dimensions
            mandala = process_frame(frame, flip_x=flip_x, flip_y=flip_y, filter_type=filter_type, distortion_type=distortion_type, time=time, speed=speed)
            mandala = cv2.cvtColor(mandala, cv2.COLOR_BGR2RGB)  # Convert to RGB for Pygame

            # Display the frame in full screen
            pygame_frame = pygame.surfarray.make_surface(mandala.swapaxes(0, 1))
            screen.blit(pygame_frame, (0, 0))
        else:
            # Attempt to find a new camera
            cap, current_camera = find_next_camera(0)
            if not cap:
                # Display "No camera found" message
                screen.fill((0, 0, 0))  # Black background
                font = pygame.font.Font(None, 74)
                text = font.render("No camera found", True, (255, 0, 0))
                text_rect = text.get_rect(center=(size[0] // 2, size[1] // 2))
                screen.blit(text, text_rect)

        pygame.display.update()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                if cap:
                    cap.release()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:  # Toggle Flip X
                    flip_x = not flip_x
                    print(f"Flip X: {flip_x}")
                elif event.key == pygame.K_y:  # Toggle Flip Y
                    flip_y = not flip_y
                    print(f"Flip Y: {flip_y}")
                elif event.key == pygame.K_c:  # Switch Camera
                    print(f"Switching from camera {current_camera}")
                    if cap:
                        cap.release()
                    cap, current_camera = find_next_camera(current_camera + 1)
                elif event.key == pygame.K_v:  # Switch Display
                    current_display += 1
                    if current_display >= num_displays:
                        current_display = 0
                    screen, size = get_fullscreen_display(current_display)
                    if screen is None:
                        print(f"Failed to switch to display {current_display}")
                    else:
                        print(f"Switched to display {current_display}")
                elif event.key == pygame.K_f:  # Cycle Filters
                    filter_type = (filter_type + 1) % 5  # Cycle through 5 filter types (0-4)
                    print(f"Filter type: {filter_type}")
                elif event.key == pygame.K_d:  # Cycle Distortions
                    distortion_type = (distortion_type + 1) % 5  # Cycle through 5 distortion types (0-4)
                    print(f"Distortion type: {distortion_type}")
                elif event.key == pygame.K_s:  # Change Speed
                    speed = (speed % 4) + 1
                    print(f"Distortion speed: {speed}")

if __name__ == "__main__":
    main()