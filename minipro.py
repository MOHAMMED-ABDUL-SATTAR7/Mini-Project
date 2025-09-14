import cv2
import mediapipe as mp
import numpy as np
import pyttsx3  # For voice feedback

# Initialize MediaPipe Face Mesh for gaze tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Camera parameters
wCam, hCam = 1280, 720  

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Change (1) to (0) if using a built-in webcam
cap.set(3, wCam)  # Set width
cap.set(4, hCam)  # Set height

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.75)
mp_draw = mp.solutions.drawing_utils

# Tip IDs for fingers (Thumb, Index, Middle, Ring, Pinky)
tipIds = [4, 8, 12, 16, 20]

# Initialize text-to-speech engine for voice feedback
engine = pyttsx3.init()

# Variables for gaze detection and voice feedback
gaze_detection_enabled = False  
voice_feedback_enabled = True  
last_gaze_direction = None  

# Variables for hand detection toggles
left_hand_detection_enabled = True
right_hand_detection_enabled = True

def is_finger_extended(lmList, tipId, jointId):
    return lmList[tipId][1] < lmList[jointId][1]  

def is_thumb_extended(lmList):
    return lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]  

def detect_gesture(fingers, hand_type):
    if hand_type in ["Right", "Left"]:
        if fingers[1] == 1 and fingers[0] == 1 and all(f == 0 for f in fingers[2:]):
            return "Finger Gun"
        elif fingers[0] == 1 and all(f == 0 for f in fingers[1:]):
            return "Thumbs Up"
        elif fingers[0] == 1 and fingers[1] == 0 and all(f == 0 for f in fingers[2:]):
            return "Thumbs Down"
        elif all(f == 0 for f in fingers):
            return "Fist"
        elif all(f == 1 for f in fingers):
            return f"{hand_type} Hand Raised"
        elif fingers == [1, 1, 0, 0, 1]:  
            return "Spidey Sign"
        elif fingers == [1, 0, 0, 0, 1]:  
            return "Call Me"
        elif fingers == [0, 1, 0, 0, 0]:  
            return "Index Finger Up"
        elif fingers == [1, 1, 1, 0, 0]:  
            return "OK Sign"
    
    return "No Gesture"

def detect_gaze(iris_landmarks, image_width):
    iris_center_x = int((iris_landmarks[0].x + iris_landmarks[1].x) / 2 * image_width)
    
    if iris_center_x < image_width / 2 - 30:
        return "Looking Left"
    elif iris_center_x > image_width / 2 + 30:
        return "Looking Right"
    else:
        return "Looking Center"

def speak(text):
    if voice_feedback_enabled:
        engine.say(text)
        engine.runAndWait()

while True:
    success, img = cap.read()
    if not success:
        continue  

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results_hand = hands.process(imgRGB)
    results_face = face_mesh.process(imgRGB)

    if gaze_detection_enabled and results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            iris_left = face_landmarks.landmark[474]
            iris_right = face_landmarks.landmark[469]

            if iris_left and iris_right:
                gaze_direction = detect_gaze([iris_left, iris_right], wCam)
                cv2.putText(img, gaze_direction, (20, hCam - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                if gaze_direction != last_gaze_direction:
                    speak(gaze_direction)
                    last_gaze_direction = gaze_direction

    if results_hand.multi_hand_landmarks:
        left_hand_info = None
        right_hand_info = None

        for hand_landmarks, hand_classification in zip(results_hand.multi_hand_landmarks, results_hand.multi_handedness):
            hand_type = hand_classification.classification[0].label  

            if (hand_type == "Left" and not left_hand_detection_enabled) or (hand_type == "Right" and not right_hand_detection_enabled):
                continue  

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lmList = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            fingers = []

            if is_thumb_extended(lmList):
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1, 5):
                if is_finger_extended(lmList, tipIds[id], tipIds[id] - 2):
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalFingers = fingers.count(1)
            if totalFingers > 5:
                totalFingers = 5

            gesture = detect_gesture(fingers, hand_type)
            print(f"Fingers Counted: {totalFingers}, Gesture: {gesture}, Hand Type: {hand_type}")

            if hand_type == "Left":
                left_hand_info = (totalFingers, gesture, hand_type)
            elif hand_type == "Right":
                right_hand_info = (totalFingers, gesture, hand_type)

        if left_hand_info:
            totalFingers, gesture, hand_type = left_hand_info
            cv2.putText(img, f"Fingers: {totalFingers}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(img, f"Gesture: {gesture}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(img, f"Hand: {hand_type}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if right_hand_info:
            totalFingers, gesture, hand_type = right_hand_info
            cv2.putText(img, f"Fingers: {totalFingers}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(img, f"Gesture: {gesture}", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(img, f"Hand: {hand_type}", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Hand and Gaze Tracking", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('w') or key == ord('W'):
        gaze_detection_enabled = not gaze_detection_enabled
        print(f"Gaze detection {'enabled' if gaze_detection_enabled else 'disabled'}.")
    elif key == ord('e') or key == ord('E'):
        voice_feedback_enabled = not voice_feedback_enabled
        print(f"Voice feedback {'enabled' if voice_feedback_enabled else 'disabled'}.")
    elif key == ord('l') or key == ord('L'):
        left_hand_detection_enabled = not left_hand_detection_enabled
        print(f"Left hand detection {'enabled' if left_hand_detection_enabled else 'disabled'}.")
    elif key == ord('r') or key == ord('R'):
        right_hand_detection_enabled = not right_hand_detection_enabled
        print(f"Right hand detection {'enabled' if right_hand_detection_enabled else 'disabled'}.")

cap.release()
cv2.destroyAllWindows()