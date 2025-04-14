import os
import imageio
import pygame
from datetime import datetime

class VideoRecorder:
    def __init__(self, save_dir="training"):
        self.save_dir = save_dir
        self.frames = []
        self.recording = False

    def start_recording(self):
        self.frames = []
        self.recording = True

    def capture_frame(self, screen):
        if self.recording:
            frame = pygame.surfarray.array3d(screen)
            frame = frame.transpose((1, 0, 2))  # Convert to (height, width, channels)
            self.frames.append(frame)

    def stop_and_save(self, video_name):
        if self.recording and self.frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_path = os.path.join(self.save_dir, timestamp)
            os.makedirs(folder_path, exist_ok=True)
            video_path = os.path.join(folder_path, f"{video_name}.mp4")
            imageio.mimsave(video_path, self.frames, fps=30)
            self.recording = False
            print(f"Video saved at: {video_path}")