# Detecting-emotion-age-gender-using-face-recognition
# Real-Time Facial Emotion, Age & Gender Detection

This project is a real-time facial analysis system built with Python, OpenCV, and deep learning. It detects **human faces** via webcam and classifies:

- **Emotions** (Happy, Sad, Angry, Surprise, Neutral, etc.)
- **Gender** (Male or Female)
- **Age Group** (e.g., 0–2, 4–6, ..., 60+)

It uses a CNN model for emotion classification and optionally leverages DeepFace for accurate age and gender prediction.

---

## Dataset

The training dataset is large (~303 MB) and available on Google Drive:

➡ **[Click here to download dataset](https://drive.google.com/uc?id=1kvWFL433YLkpuMqx4iS06zlySMQN8hVW&export=download)**

After downloading, extract the dataset and place it in the project’s `dataset/` folder.

---

## Technologies Used

- **Python 3.10+**
- **OpenCV**
- **MediaPipe**
- **TensorFlow / Keras**
- **DeepFace (optional for age/gender)**
- NumPy, Pandas, Matplotlib

---

## 1. Installation & Setup

1. **Clone this repo**
   ```bash
   git clone https://github.com/your-username/emotion-age-gender-detector.git
   cd emotion-age-gender-detector

## 2. Create virtual environment
   python3 -m venv venv
   source venv/bin/activate  
   # for macOS/Linux
   # OR
   venv\Scripts\activate     
   # for Windows

## 3. Install Dependencies
   pip install -r requirements.txt

## 4. Download & unzip dataset
   Download from Google Drive
   Place the unzipped contents into the dataset/ folder

## 5. Run the Application


