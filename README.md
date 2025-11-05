# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows  

## Program
```
import numpy as np
import cv2 
import matplotlib.pyplot as plt
```
```
model = cv2.imread('image_01.png',0)
withglass = cv2.imread('image_02.png',0)
group = cv2.imread('image_03.png',0)
```
```
plt.figure(figsize=(20,10))
plt.subplot(131);plt.imshow(cv2.resize(model, (1000, 1000)),cmap='gray');plt.title("Model")
plt.subplot(132);plt.imshow(cv2.resize(withglass, (1000, 1000)),cmap='gray');plt.title("Model with glass")
plt.subplot(133);plt.imshow(cv2.resize(group, (1000, 1000)),cmap='gray');plt.title("Group")
plt.show()
```
```
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```
```
def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.1, minNeighbors=5) 
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
    return face_img
```
```
result = detect_face(withglass)
plt.imshow(result,cmap='gray')
plt.show()
```
```
result = detect_face(model)
plt.imshow(result,cmap='gray')
plt.show()
```
```
result = detect_face(group)
plt.imshow(result,cmap='gray')
plt.show()
```
```
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
```
```
def detect_eyes(img):
    face_img = img.copy()
    eyes = eye_cascade.detectMultiScale(face_img) 
    for (x,y,w,h) in eyes: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
    return face_img
```
```
result = detect_eyes(model)
plt.imshow(result,cmap='gray')
plt.show()
```
```
result = detect_eyes(withglass)
plt.imshow(result,cmap='gray')
plt.show()
```
```
result = detect_eyes(group)
plt.imshow(result,cmap='gray')
plt.show()
```
```
import cv2
import matplotlib.pyplot as plt

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function for face detection
def detect_face(img):
    if img is None:
        return None  # Prevents AttributeError if frame is invalid

    face_img = img.copy()
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around faces
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 3)

    return face_img


# --- Webcam capture ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Could not open camera. If you're using Google Colab, upload a photo or video file instead.")
else:
    # Set up matplotlib
    plt.ion()
    fig, ax = plt.subplots()

    ret, frame = cap.read()
    if not ret:
        print(" Could not read first frame from camera.")
    else:
        frame = detect_face(frame)
        im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Video Face Detection')

        while True:
            ret, frame = cap.read()
            if not ret:
                print(" Frame not received. Exiting loop.")
                break

            frame = detect_face(frame)
            if frame is None:
                print(" Skipping invalid frame.")
                continue

            # Update the image in matplotlib window
            im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.pause(0.1)

    cap.release()
    plt.close()
```
## Output
<img width="1334" height="433" alt="image" src="https://github.com/user-attachments/assets/c4b60489-8797-413e-af15-6501b4c69a8a" />
<img width="491" height="467" alt="image" src="https://github.com/user-attachments/assets/d255bb40-7ef7-4c6e-88fc-688a3716adce" />
<img width="386" height="468" alt="image" src="https://github.com/user-attachments/assets/18ab1d4c-cb0a-45ac-9412-bd720b48368a" />
<img width="625" height="393" alt="image" src="https://github.com/user-attachments/assets/8133884b-0164-4106-bd92-17b40cdb1084" />
<img width="379" height="466" alt="image" src="https://github.com/user-attachments/assets/e8d08008-7818-4f85-a615-6190b55dcc46" />
<img width="490" height="465" alt="image" src="https://github.com/user-attachments/assets/928db8d4-6ed1-49d3-bff5-6748ca34c3ad" />
<img width="631" height="400" alt="image" src="https://github.com/user-attachments/assets/8ae48541-b700-4d2d-8ce3-33c640612f8d" />

