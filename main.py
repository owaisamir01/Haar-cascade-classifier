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

# Specify the file path (image or video)
file_path = "smile.jpg"  # Replace with your image or video file

# Check if the file exists
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    # Check file extension to decide processing
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

                # Region of interest (ROI) for eyes and smiles
                roi_gray = gray[y:y + height, x:x + width]
                roi_color = frame[y:y + height, x:x + width]

                # Detect eyes in the ROI
                eyes = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 20),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                # Detect smiles in the ROI
                smiles = smile_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.7,
                    minNeighbors=20,
                    minSize=(25, 25),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

            # Resize the frame
            resized_frame = cv2.resize(frame, (640, 480))  # Resize to 640x480

            cv2.imshow("Video Frame - Faces, Eyes, and Smiles", resized_frame)

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

                # Region of interest (ROI) for eyes and smiles
                roi_gray = gray[y:y + height, x:x + width]
                roi_color = image[y:y + height, x:x + width]

                # Detect eyes in the ROI
                eyes = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 20),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                # Detect smiles in the ROI
                smiles = smile_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.7,
                    minNeighbors=20,
                    minSize=(25, 25),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

            # Resize the image
            resized_image = cv2.resize(image, (640, 480))  # Resize to 640x480

            cv2.imshow("Image - Faces, Eyes, and Smiles", resized_image)
            cv2.waitKey(0)  # Wait for a key press to close the window

    else:
        print("Error: Unsupported file format.")

cv2.destroyAllWindows()
