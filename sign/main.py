import pickle
import cv2
import mediapipe as mp
import numpy as np

# jkj

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the width and height of the captured frame (for example, 1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set frame width to 1280 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set frame height to 720 pixels

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label mapping (assuming labels are integers)
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'L', 6: 'Hand', 7: 'space', 8: 'Y', 9: 'W', 10: 'X', 11: 'T'}

def normalize_landmarks(hand_landmarks, x_min, y_min):
    """Normalize hand landmarks with respect to the given minimum x and y."""
    data = []
    for i in range(len(hand_landmarks)):
        x = hand_landmarks[i].x
        y = hand_landmarks[i].y
        data.append(x - x_min)  # Normalize x-coordinate
        data.append(y - y_min)  # Normalize y-coordinate
    return data

while True:
    data_aux = []

    ret, frame = cap.read()

    # Get frame dimensions for later use (bounding box drawing)
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Only process the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]  # Take the first hand only
        
        # Draw the landmarks for the first hand
        mp_drawing.draw_landmarks(
            frame, hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Extract x and y coordinates for the first hand
        x = [landmark.x for landmark in hand_landmarks.landmark]
        y = [landmark.y for landmark in hand_landmarks.landmark]

        # Normalize the hand's landmarks with its min x and y values
        data_aux = normalize_landmarks(hand_landmarks.landmark, min(x), min(y))

        # Ensure the feature vector has the correct length (42)
        if len(data_aux) == 42:  # This should always be true for one hand
            # Make the prediction using the model
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Draw the bounding box and predicted label for the hand
            x1 = int(min(x) * W) - 10
            y1 = int(min(y) * H) - 10
            x2 = int(max(x) * W) - 10
            y2 = int(max(y) * H) - 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            # Check if the 'w' key is pressed and the predicted character matches any label
            key = cv2.waitKey(1) & 0xFF
            if key == ord('w') and predicted_character in labels_dict.values():
                print(f"Key 'w' pressed. Matching label: {predicted_character}")

    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII value for the ESC key
        break

cap.release()
cv2.destroyAllWindows()
