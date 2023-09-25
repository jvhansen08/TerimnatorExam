import matplotlib.pyplot as plt
import psutil
import time
import matplotlib
import cv2
import numpy as np
from fer import FER
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
matplotlib.use('TkAgg')

if __name__ == '__main__':
    # Function to calculate frames per second (FPS)
    def calculate_fps(start_time, end_time, frame_count):
        elapsed_time = end_time - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            return fps
        else:
            return 0.0

    print("Press q on the output window to stop")
    print("Use live webcam feed or pre-recorded video file? (1 for webcam, 2 for video file)")
    choice = input()

    # Load your pre-trained emotion recognition model from the FER library
    emotion_detector = FER()

    # Initialize the video capture
    if choice == '1':
        video_capture = cv2.VideoCapture(0)  # Webcam
    else:
        video_capture = cv2.VideoCapture('./emotionRecording.mp4')  # Video file

    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()

    # Initialize empty lists to store emotion predictions
    emotions = []

    while True:
        if video_capture.isOpened():
            break

    while True:
        # Read a frame from the video source
        ret, frame = video_capture.read()

        if not ret:
            break

        # Perform emotion recognition on the frame
        emotions_result = emotion_detector.detect_emotions(frame)

        # Append the emotions result to the list
        emotions.append(emotions_result)

        frame_count += 1

        # Display the frame with emotions
        for result in emotions_result:
            if result:
                x, y, w, h = result["box"]
                emotion_dict = result["emotions"]
                emotion_label = max(emotion_dict, key=emotion_dict.get)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Emotion Recognition', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate and print FPS and CPU usage
    end_time = time.time()
    fps = calculate_fps(start_time, end_time, frame_count)
    cpu_usage = psutil.cpu_percent()

    print(f"Frames analyzed: {frame_count}")
    print(f"FPS: {fps:.2f}")
    print(f"CPU Usage: {cpu_usage:.2f}%")

    # Convert emotion labels to strings for plotting
    emotions_labels = []

    for result in emotions:
        if result:
            emotion_label = max(result[0]["emotions"], key=result[0]["emotions"].get)
            emotions_labels.append(emotion_label)
        else:
            emotions_labels.append(None)

    # Release the video capture and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

    # Replace None values with a placeholder string
    emotions_labels = [label if label is not None else 'No Emotion Detected' for label in emotions_labels]

    # Plot emotions over time (if there are any emotions)
    if emotions_labels:
        time_axis = np.arange(len(emotions_labels))
        plt.figure(figsize=(10, 5))
        plt.scatter(time_axis, emotions_labels, marker='o', c='b', label='Emotions', s=50)
        plt.xlabel('Time (frames)')
        plt.ylabel('Emotion')
        plt.legend()
        plt.title('Emotion Recognition Over Time')
        plt.show()
    else:
        print("No emotions detected.")