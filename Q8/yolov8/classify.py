from ultralytics import YOLO
import time
import psutil
import threading
import os

def monitor_cpu_usage(interval, cpu_percentages):
    while not stop_monitoring:
        cpu_percentages.append(psutil.cpu_percent(interval=interval))

# Load a model
model = YOLO("yolov8x-cls.pt")  # load a pretrained model

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
model.predict(source="./grocerystore/", project='./ultralytics', name='classifications', save=True)

# End measuring time
end = time.time()

# Stop the CPU monitoring thread
stop_monitoring = True
cpu_monitoring_thread.join()
#photo count for fps
path, dirs, files = next(os.walk("./ultralytics/classifications"))
file_count = len(files)
fps = file_count/(end-start)

print("FPS: {:.2f}".format(fps))

# Calculate average CPU usage
average_cpu_usage = sum(cpu_percentages) / len(cpu_percentages)
print("CPU usage: {:.2f}%".format(average_cpu_usage))