import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Open camera
cam = cv2.VideoCapture(0)
hands = mp_hands.Hands()
while True:
    success, image = cam.read()
    if not success:
        break
    # Flip and convert the image
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image
    results = hands.process(image_rgb)
    # Check if hands were detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    # Show the output
    cv2.imshow("HandTracker", image)

    # Exit on 'q' or ESC
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break
cam.release()