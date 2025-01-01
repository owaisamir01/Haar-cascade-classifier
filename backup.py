import cv2
import os
import pathlib

# Load Haar Cascade XML files for face, eyes, and smile
face_cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
eye_cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_eye.xml"
smile_cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_smile.xml"

face_cascade = cv2.CascadeClassifier(str(face_cascade_path))
eye_cascade = cv2.CascadeClassifier(str(eye_cascade_path))
smile_cascade = cv2.CascadeClassifier(str(smile_cascade_path))

# File path (image or video)
file_path = "test video.mp4"  # Replace with your image or video file

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video extensions
        camera = cv2.VideoCapture(file_path)
        while True:
            ret, frame = camera.read()
            if not ret:
                print("End of video or unable to fetch frame.")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, width, height) in faces:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)
                confidence_face = (width * height) / (gray.shape[0] * gray.shape[1]) * 100
                cv2.putText(frame, f"Face: {confidence_face:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Region of interest for the face
                roi_gray = gray[y:y + height, x:x + width]
                roi_color = frame[y:y + height, x:x + width]

                # Detect smiles in the lower half of the face
                mouth_roi_gray = roi_gray[height // 2:, :]
                smiles = smile_cascade.detectMultiScale(
                    mouth_roi_gray,
                    scaleFactor=1.8,  # Fine-tune for better accuracy
                    minNeighbors=20,  # Increase for fewer false positives
                    minSize=(25, 25),  # Minimum size for a valid smile
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, height // 2 + sy), (sx + sw, height // 2 + sy + sh), (0, 0, 255), 2)
                    confidence_smile = (sw * sh) / (roi_gray.shape[0] * roi_gray.shape[1]) * 100
                    cv2.putText(roi_color, f"Smile: {confidence_smile:.2f}%", (sx, height // 2 + sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            resized_frame = cv2.resize(frame, (640, 480))
            cv2.imshow("Video Frame - Faces and Smiles", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()

    elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):  # Image extensions
        image = cv2.imread(file_path)
        if image is None:
            print("Error: Unable to load the image. Check the file format.")
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, width, height) in faces:
                cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 0), 2)
                confidence_face = (width * height) / (gray.shape[0] * gray.shape[1]) * 100
                cv2.putText(image, f"Face: {confidence_face:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Detect smiles
                mouth_roi_gray = gray[y + height // 2:y + height, x:x + width]
                smiles = smile_cascade.detectMultiScale(
                    mouth_roi_gray,
                    scaleFactor=1.8,
                    minNeighbors=20,
                    minSize=(25, 25),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(image, (x + sx, y + height // 2 + sy), (x + sx + sw, y + height // 2 + sy + sh), (0, 0, 255), 2)
                    confidence_smile = (sw * sh) / (gray.shape[0] * gray.shape[1]) * 100
                    cv2.putText(image, f"Smile: {confidence_smile:.2f}%", (x + sx, y + height // 2 + sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            resized_image = cv2.resize(image, (640, 480))
            cv2.imshow("Image - Faces and Smiles", resized_image)
            cv2.waitKey(0)

cv2.destroyAllWindows()
