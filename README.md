# Face Detection and Recognition App

This project is a Streamlit-based web application for face detection and recognition using various models including Haar Cascade, ArcFace.

## Features

- **Face Detection and Recognition**:
  - Using Haar Cascade
  - Using ArcFace (Deep Learning-based)
- **Webcam Integration**:
  - Real-time face detection and recognition using the webcam

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/kaushalraja/face-detection-recognition-app.git
    cd face-detection-recognition-app
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:
    ```sh
    streamlit run src/app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Use the interface to upload images for face detection and recognition or turn on the webcam for real-time detection.

## Adding Known Faces

1. Expand the "Add a Known Face" section.
2. Enter the name of the person.
3. Upload an image of the person.
4. Click "Add Face to Database" to save the face encoding to the database.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [insightface](https://github.com/deepinsight/insightface)
