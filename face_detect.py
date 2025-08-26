import cv2
import os
import face_recognition

# ---------------- Load known faces ----------------
known_encodings = []
known_names = []

KNOWN_DIR = "Ai-personal-assistant\known_faces"

for filename in os.listdir(KNOWN_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(KNOWN_DIR, filename)
        img = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(img)
        if encs:
            known_encodings.append(encs[0])
            known_names.append(os.path.splitext(filename)[0])
            print(f"Loaded: {filename}")

# ---------------- Start Webcam ----------------
cap = cv2.VideoCapture(0)
TOLERANCE = 0.5   # lower = stricter

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB (required for face_recognition)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces & encodings
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    for (top, right, bottom, left), enc in zip(locations, encodings):
        name = "Unknown"
        distances = face_recognition.face_distance(known_encodings, enc)
        if len(distances) > 0:
            best_idx = distances.argmin()
            if distances[best_idx] < TOLERANCE:
                name = known_names[best_idx]

        # Draw results
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
