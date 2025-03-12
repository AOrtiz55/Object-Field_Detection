import cv2
import depthai as dai
import numpy as np
import time

# Distance thresholds in millimeters
SAFE_DISTANCE = 450  # 45 cm
CAUTION_START = 380  # 38 cm
DANGER_DISTANCE = 370  # 37 cm
ALERT_DISTANCE = 2000  # 100 cm (1 meter)

# Initialize flashing alert settings
flash_alert = False
last_flash_time = time.time()

# Create DepthAI pipeline
pipeline = dai.Pipeline()

# Create color camera
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setFps(30)

# Create depth camera
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align depth to RGB
stereo.setSubpixel(True)  # Improve depth accuracy

# Outputs
rgbOut = pipeline.create(dai.node.XLinkOut)
rgbOut.setStreamName("rgb")
camRgb.video.link(rgbOut.input)

depthOut = pipeline.create(dai.node.XLinkOut)
depthOut.setStreamName("depth")
stereo.depth.link(depthOut.input)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# Start the OAK-D Lite pipeline
with dai.Device(pipeline) as device:
    rgbQueue = device.getOutputQueue("rgb", maxSize=1, blocking=False)
    depthQueue = device.getOutputQueue("depth", maxSize=1, blocking=False)

    while True:
        inRgb = rgbQueue.get()
        inDepth = depthQueue.get()

        if inRgb is None or inDepth is None:
            continue  # Skip if no frame received

        # Convert frame to OpenCV format
        frame = inRgb.getCvFrame()
        depthFrame = inDepth.getFrame()

        # Resize depth frame to match RGB resolution
        depthFrame = cv2.resize(depthFrame, (frame.shape[1], frame.shape[0]))

        # Apply median blur to reduce noise
        depthFrame = cv2.medianBlur(depthFrame, 5)

        # Create a mask to detect objects within 1 meter
        mask = (depthFrame > 100) & (depthFrame < ALERT_DISTANCE)
        mask = mask.astype(np.uint8) * 255

        # Find contours for detected objects
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Flashing alert logic
        current_time = time.time()
        if current_time - last_flash_time > 1:  # Toggle every 1 second
            flash_alert = not flash_alert
            last_flash_time = current_time

        # Store minimum distance of all detected objects
        min_distance = np.inf  # Start with an infinite value
        object_too_close = False  # Track if any object is too close
        show_only_red = False  # If any object is under 37 cm, set this flag to True

        # Process detected objects
        object_dots = []  # Store dots to draw later

        for contour in contours:
            contour_distances = []
            object_has_danger_point = False  # Track if any point in the object is < 37 cm

            # Collect distances for each object
            for point in contour:
                x, y = point[0]
                depth_value = depthFrame[y, x]

                if depth_value > 0:  # Ignore invalid depth readings
                    contour_distances.append(depth_value)

                    # If any part of the object is too close, set flag
                    if depth_value < DANGER_DISTANCE:
                        object_has_danger_point = True
                        show_only_red = True  # Force only red dots to be shown

            # Get the closest point in this contour
            if contour_distances:
                contour_min_distance = min(contour_distances)
                min_distance = min(min_distance, contour_min_distance)

                # Determine color for this object
                if object_has_danger_point:
                    dot_color = (0, 0, 255)  # Red
                    object_too_close = True
                elif CAUTION_START <= contour_min_distance <= SAFE_DISTANCE:
                    dot_color = (0, 255, 255)  # Yellow
                else:
                    dot_color = (0, 255, 0)  # Green

                # Store dots for drawing
                for point in contour:
                    x, y = point[0]
                    object_dots.append((x, y, dot_color))

        # Draw only red dots if an object is too close
        for x, y, color in object_dots:
            if show_only_red:
                if color == (0, 0, 255):  # Only draw red dots if an object is too close
                    cv2.circle(frame, (x, y), 2, color, -1)
            else:
                cv2.circle(frame, (x, y), 2, color, -1)  # Draw all dots normally

        # Flashing Alert Message
        if object_too_close and flash_alert:
            cv2.putText(frame, "WARNING: OBJECT TOO CLOSE!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Display the **minimum** detected distance
            min_distance_text = f"Object Distance: {min_distance / 10:.1f} cm"
            cv2.putText(frame, min_distance_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Resize the frame for smaller display (720x500)
        resized_frame = cv2.resize(frame, (720, 500))

        # Show results
        cv2.imshow("OAK-D Lite - Object Proximity Detection", resized_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
