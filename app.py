# Install dependencies before running:
# pip install streamlit opencv-python faiss-cpu torch torchvision torchaudio ultralytics insightface onnxruntime

import streamlit as st
import cv2
import os
import numpy as np
import faiss
import tempfile
import zipfile
from PIL import Image
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ---------- CONFIGURATION ----------
EMBEDDING_DIM = 512
THRESHOLD = 0.3  # Cosine similarity threshold

app = FaceAnalysis(name='buffalo_l', root='/tmp')
app.prepare(ctx_id=-1, det_size=(640, 640))  # Use CPU for inference

def get_face_embedding(image, source=""):
    faces = app.get(image)
    if not faces:
        st.warning(f"No face detected in: {source}")
    elif len(faces) > 1:
        st.warning(f"Multiple faces detected in: {source}")
    return faces[0].embedding.astype('float32') if faces else None

def load_dataset_embeddings(dataset_dir):
    embeddings = []
    names = []
    for file in os.listdir(dataset_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(dataset_dir, file)
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            embedding = get_face_embedding(img_rgb, source=path)
            if embedding is not None:
                embeddings.append(embedding)
                names.append(file.rsplit(".", 1)[0])
            else:
                st.warning(f"Failed to generate embedding for {file}")
    st.info(f"✅ Processed {len(names)} valid face images from dataset.")
    return np.array(embeddings).astype('float32'), names

def build_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Cosine similarity
    index.add(embeddings)
    return index

def match_face(query_embedding, index, names):
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    D, I = index.search(query_embedding.reshape(1, -1), k=1)
    if D[0][0] > THRESHOLD:
        return names[I[0][0]], D[0][0]
    return "Unknown", D[0][0]

# ----------- STREAMLIT UI -----------
st.set_page_config(page_title="Face Recognition System", layout="centered")
st.title("🔍 Real-Time Face Recognition with InsightFace & FAISS")

uploaded_zip = st.file_uploader("Upload a ZIP file of face images (dataset)", type="zip")

dataset_dir = None
if uploaded_zip:
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        candidate_dirs = [os.path.join(temp_dir, d) for d in os.listdir(temp_dir)]
        dataset_dir = None
        for d in candidate_dirs:
            if os.path.isdir(d):
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(d)):
                    dataset_dir = d
                    break

        if not dataset_dir:
            st.error("No images found in the uploaded ZIP. Please ensure it contains face images.")
            st.stop()

        with st.spinner("Loading face embeddings from uploaded dataset..."):
            embeddings, names = load_dataset_embeddings(dataset_dir)
            if len(embeddings) == 0:
                st.error("No valid face embeddings found in dataset.")
                st.stop()
            index = build_faiss_index(embeddings)
        st.success("Embeddings loaded from uploaded dataset.")
else:
    st.warning("Please upload a zipped dataset to proceed.")
    st.stop()

option = st.radio("Choose Input Method", ("📷 Webcam", "📁 Upload Image"))

if option == "📷 Webcam":
    st.info("📸 Use your webcam to capture a photo for recognition.")
    picture = st.camera_input("Capture your face")

    if picture is not None:
        img = Image.open(picture).convert("RGB")
        img_np = np.array(img)
        st.image(img_np, caption="Captured Image", use_column_width=True)

        faces = app.get(img_np)
        if faces:
            embedding = faces[0].embedding.astype('float32')
            name, dist = match_face(embedding, index, names)
            st.success(f"Recognized as: {name} (Similarity: {dist:.2f})")
        else:
            st.warning("No face detected in the captured photo.")

elif option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"], key="upload-image")
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        faces = app.get(img_np)
        if faces:
            embedding = faces[0].embedding.astype('float32')
            name, dist = match_face(embedding, index, names)
            st.success(f"Recognized as: {name} (Similarity: {dist:.2f})")
        else:
            st.warning("No face detected in the uploaded image.")
