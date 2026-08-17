# 🤟 Real-Time Sign Language Detection System

An interactive computer vision and machine learning application that detects hand gestures and translates them into corresponding sign language words. Built with Python, OpenCV, Google MediaPipe, and Scikit-Learn, all deployed inside a modern Streamlit web interface.

---

## 📌 Features

- **Dual Detection Modes**:
  - **Upload Image**: Process static images (`.jpg`, `.jpeg`, `.png`) to identify hand signs.
  - **Real-Time Video**: Continuous, low-latency sign detection directly from your webcam.
- **21 3D Landmark Extraction**: Leverages Google MediaPipe to track 21 distinct hand landmarks across 3-dimensional space (63 feature inputs).
- **Machine Learning Classification**: Trained using a Random Forest Classifier to categorize hand gestures.
- **Time-Gated Access Control**: Built-in operational scheduling that automatically enables functionality during specific operating hours (e.g., 6:00 PM – 10:00 PM) and locks outside this window.
- **Interactive Web GUI**: Clean, responsive frontend built using Streamlit.

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Computer Vision**: [OpenCV](https://opencv.org/)
- **Landmark Estimation**: [Google MediaPipe](https://developers.google.com/mediapipe)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (Random Forest)
- **Data & Image Processing**: NumPy, Pillow

---

## 📁 Project Structure

```text
├── app.py                  # Main Streamlit application
├── collect_data.py         # (Optional) Data collection script for custom gestures
├── train_model.py          # (Optional) Model training & export script
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
