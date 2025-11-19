import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# Function to detect emotion from landmarks
def detect_emotion(landmarks):
    # Landmarks for mouth
    top_lip = landmarks[13].y
    bottom_lip = landmarks[14].y

    mouth_open = bottom_lip - top_lip

    # Landmarks for eyebrows
    left_brow = landmarks[65].y
    left_eye = landmarks[159].y
    brow_raise = left_brow - left_eye

    # Simple logic
    if mouth_open > 0.03:
        return "Happy 🙂"
    elif brow_raise < -0.01:
        return "Sad 🙁"
    else:
        return "Neutral 😐"

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION
                )

                landmarks = face_landmarks.landmark
                emotion = detect_emotion(landmarks)

                # Show text on screen
                cv2.putText(frame, f"Emotion: {emotion}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
