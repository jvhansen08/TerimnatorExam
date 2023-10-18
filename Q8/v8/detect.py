from ultralytics import YOLO
import time
import psutil
import threading

def monitor_cpu_usage(interval, cpu_percentages):
    while not stop_monitoring:
        cpu_percentages.append(psutil.cpu_percent(interval=interval))

# Load a model
model = YOLO("yolov8x.pt")  # load a pretrained model

# Initialize variables for CPU usage tracking
cpu_percentages = []
stop_monitoring = False

# Set the interval for CPU usage sampling (in seconds)
sampling_interval = 1  # You can adjust this value as needed

# Start the CPU monitoring thread
cpu_monitoring_thread = threading.Thread(target=monitor_cpu_usage, args=(sampling_interval, cpu_percentages))
cpu_monitoring_thread.start()

# Begin measuring time
start = time.time()

# Use the model to detect objects
model.predict(source="./grocerystore/", project='./ultralytics', name='detections', save=True)
