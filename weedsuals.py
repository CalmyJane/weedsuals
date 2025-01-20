import cv2
import pygame
import sys
import numpy as np

def initialize_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)  # Initialize the camera capture
    if not cap.isOpened():
        return None
    return cap

def get_fullscreen_display():
    pygame.init()
    info = pygame.display.Info()  # Get the current display info
    size = (info.current_w, info.current_h)  # Fullscreen size
    screen = pygame.display.set_mode(size, pygame.FULLSCREEN)  # Set to fullscreen
    return screen, size

def process_frame(frame, flip_x=False, flip_y=False):
    # Apply optional flips
    if flip_x:
        frame = cv2.flip(frame, 1)  # Horizontal flip
    if flip_y:
        frame = cv2.flip(frame, 0)  # Vertical flip

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
    cap, current_camera = find_next_camera(current_camera)

    screen, size = get_fullscreen_display()
    flip_x = False  # Initial state for Flip X
    flip_y = False  # Initial state for Flip Y

    while True:
        if cap:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Switching to next camera.")
                cap.release()
                cap, current_camera = find_next_camera(current_camera + 1)
                continue

            # Resize the camera feed to fit 1/4th of the screen
            frame = cv2.resize(frame, (size[0] // 2, size[1] // 2))  # Adjust frame dimensions
            mandala = process_frame(frame, flip_x=flip_x, flip_y=flip_y)
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

if __name__ == "__main__":
    main()
