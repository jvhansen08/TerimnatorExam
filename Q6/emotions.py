import cv2
import numpy as np
from fer import FER
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend

import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Load your pre-trained emotion recognition model from the FER library
    emotion_detector = FER()

    # Load your video files using OpenCV
    video1 = cv2.VideoCapture('./Video_One.mp4')
    video2 = cv2.VideoCapture('./Video_Two.mp4')

    # Initialize empty lists to store emotion predictions
    emotions_video1 = np.array([])
    emotions_video2 = np.array([])

    while True:
        # Read frames from the videos
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        if not ret1 or not ret2:
            break

        # Perform emotion recognition on the frames
        emotions1 = emotion_detector.top_emotion(frame1)
        emotions2 = emotion_detector.top_emotion(frame2)

        emotions_video1 = np.append(emotions_video1, emotions1)
        emotions_video2 = np.append(emotions_video2, emotions2)

    # Create a time axis for plotting
    time_axis = np.arange(len(emotions_video1))


    # Plot emotions over time for video 1
    plt.figure(figsize=(10, 10))
    plt.scatter(time_axis, emotions_video1, marker='o', c='b', label='Video 1 Emotions', s=50)  # 'o' for circle markers
    plt.xlabel('Time (frames)')
    plt.ylabel('Emotion')
    plt.legend()
    plt.title('Emotion Recognition for Video 1')
    plt.show()

    # Plot emotions over time for video 2
    plt.figure(figsize=(10, 10))
    plt.scatter(time_axis, emotions_video2, marker='o', c='b', label='Video 2 Emotions', s=50)  # 'o' for circle markers
    plt.xlabel('Time (frames)')
    plt.ylabel('Emotion')
    plt.legend()
    plt.title('Emotion Recognition for Video 2')
    plt.show()
