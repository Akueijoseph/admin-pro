import cv2
import face_recognition
from flask import Flask, request, jsonify

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

app = Flask(__name__)

@app.route('/access-control', methods=['POST'])
def access_control():
    # Retrieve the video frame from the request
    file = request.files['video']
    file_bytes = file.read()

    # Convert the video frame to numpy array
    nparr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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

    # Return access control result as JSON response
    response = {
        'accessGranted': access_granted
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()