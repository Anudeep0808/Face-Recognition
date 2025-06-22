# 🔍 Real-Time Face Recognition with InsightFace & FAISS

This project is a web-based face recognition system built using **Streamlit**, **InsightFace**, and **FAISS**. It supports both real-time webcam photo capture and image upload, compares face embeddings, and returns the closest match from your custom uploaded dataset.

## 🚀 Live Demo

👉 Try the deployed app here:  
[https://face-recognition-faiss.streamlit.app/](https://face-recognition-faiss.streamlit.app/)

## 📁 Features

- Upload a ZIP file containing labeled face images (`.jpg`, `.jpeg`, `.png`)
- Capture photo from webcam or upload an image to recognize
- Real-time recognition using pre-trained `buffalo_l` model from InsightFace
- Embedding comparison using **Cosine Similarity** via FAISS
- Built entirely in Python, deployable via Streamlit Cloud


