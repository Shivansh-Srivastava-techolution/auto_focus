import os
import sys
import cv2
import time
import threading

# Global state
best_focus_value = 128
best_focus_metric = 0
stop_optimization = False
roi = None
points = []
optimization_thread = None

def decode_fourcc(value):
    return "".join([chr((value >> 8 * i) & 0xFF) for i in range(4)])

def configure_camera(cap, width=8000, height=6000, fps=5, codec="MJPG"):
    if not cap or not cap.isOpened():
        return None
    fourcc = cv2.VideoWriter_fourcc(*codec)
    old_fourcc = decode_fourcc(int(cap.get(cv2.CAP_PROP_FOURCC)))
    if cap.set(cv2.CAP_PROP_FOURCC, fourcc):
        print(f"Codec changed from {old_fourcc} to {decode_fourcc(int(cap.get(cv2.CAP_PROP_FOURCC)))}")
    else:
        print(f"Error: Could not change codec from {old_fourcc}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    print(f"Camera configured with FPS: {cap.get(cv2.CAP_PROP_FPS)}, "
          f"Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, "
          f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    return cap

def get_frame(camera, focus_value):
    camera.set(cv2.CAP_PROP_FOCUS, focus_value)
    time.sleep(0.05)
    for _ in range(2):
        camera.read()
    ret, frame = camera.read()
    return frame if ret else None

def calculate_focus_metric(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def evaluate_focus(frame, roi, resize_scale=0.75):
    x, y, w, h = roi
    roi_frame = frame[y:y+h, x:x+w]
    
    new_width = max(1, int(w * resize_scale))
    new_height = max(1, int(h * resize_scale))    
    resized_roi = cv2.resize(roi_frame, (new_width, new_height))

    return calculate_focus_metric(resized_roi)

def fibonacci_search_focus(camera, roi):
    global best_focus_value, best_focus_metric, stop_optimization
    low, high = 50, 100
    fib = [1, 1]
    while fib[-1] < (high - low):
        fib.append(fib[-1] + fib[-2])
    n = len(fib) - 1
    x1 = low + fib[n-2] * (high - low) / fib[n]
    x2 = low + fib[n-1] * (high - low) / fib[n]
    frame1 = get_frame(camera, x1)
    frame2 = get_frame(camera, x2)
    fx1 = evaluate_focus(frame1, roi)
    fx2 = evaluate_focus(frame2, roi)
    while n > 1:
        if fx1 > fx2:
            high, x2, fx2 = x2, x1, fx1
            x1 = low + fib[n-3] * (high - low) / fib[n-1]
            frame1 = get_frame(camera, x1)
            fx1 = evaluate_focus(frame1, roi)
        else:
            low, x1, fx1 = x1, x2, fx2
            x2 = low + fib[n-2] * (high - low) / fib[n-1]
            frame2 = get_frame(camera, x2)
            fx2 = evaluate_focus(frame2, roi)
        n -= 1
    final_focus = int((low + high) / 2)
    frame = get_frame(camera, final_focus)
    final_metric = evaluate_focus(frame, roi)
    best_focus_value = final_focus
    best_focus_metric = final_metric
    camera.set(cv2.CAP_PROP_FOCUS, final_focus)
    stop_optimization = True
    print(f"Best Focus Value: {best_focus_value}, Metric: {best_focus_metric}")

def mouse_callback(event, x, y, flags, param):
    global points, roi
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
        if len(points) == 2:
            x0, y0 = points[0]
            x1, y1 = points[1]
            x_min, x_max = min(x0, x1), max(x0, x1)
            y_min, y_max = min(y0, y1), max(y0, y1)
            roi = (x_min, y_min, x_max - x_min, y_max - y_min)

def manual_select_roi(frame):
    global points, roi
    points = []
    roi = None
    clone = frame.copy()
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select ROI", mouse_callback)
    while True:
        temp = clone.copy()
        for p in points:
            cv2.circle(temp, p, 10, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.rectangle(temp, points[0], points[1], (0, 255, 0), 5)
        cv2.putText(temp, "Click two points to select ROI. Press 's' to start.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Select ROI", temp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and roi:
            break
        elif key == ord('q'):
            points = []
            roi = None
            break
        elif key == ord('r'):
            points = []
            roi = None
    cv2.destroyWindow("Select ROI")
    return roi

def main():
    global roi, stop_optimization, optimization_thread
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap = configure_camera(cap, width=4000, height=3000, fps=15, codec="MJPG")
    if not cap:
        print("Camera config failed.")
        return 
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if not cap.isOpened():
        print("Camera error.")
        return
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        return
    roi = manual_select_roi(frame)
    if not roi:
        print("No ROI selected.")
        return
    cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Feed", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not stop_optimization and (optimization_thread is None or not optimization_thread.is_alive()):
            start_time = time.perf_counter()
            def run_optimization():
                fibonacci_search_focus(cap, roi)
                stop_time = time.perf_counter()
                print("Focus Time", stop_time - start_time)
            optimization_thread = threading.Thread(target=run_optimization)
            optimization_thread.start()
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Best Focus: {best_focus_value}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(frame, f"Optimized: {stop_optimization}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("Live Feed", frame)
        key = cv2.waitKey(1)
        if key == 27:
            print("Exiting...")
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
