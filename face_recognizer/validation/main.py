import cv2
import face_recognition

# Load images and encode known faces
known_face_encodings = []
known_face_names = []

# Add known faces
known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file("akuei_kuol.jpg"))[0])
known_face_names.append("Akuei Kuol")
known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file("bol_mayuek.jpg"))[0])
known_face_names.append("Bol Mayuek")
known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file("juuk_kuol.jpg"))[0])
known_face_names.append("Juuk Kuol")

# Initialize video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to RGB color
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Initialize access control flag
    access_granted = False

    # Loop through each face in the current frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # Check if there is a match
        if True in matches:
            access_granted = True
            break

    # Draw rectangle and access status on the frame
    if access_granted:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, "Access Granted", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    else:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, "Access Denied", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()