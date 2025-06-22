import streamlit as st
import cv2
import os
import numpy as np
import faiss
import tempfile
from PIL import Image
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ---------- CONFIGURATION ----------
DATASET_DIR = "faces_dataset"  
EMBEDDING_DIM = 512
THRESHOLD = 0.3  

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(640, 640))  

def get_face_embedding(image, source=""):
    faces = app.get(image)
    if not faces:
        st.warning(f"No face detected in: {source}")
    elif len(faces) > 1:
        st.warning(f"Multiple faces detected in: {source}")
    return faces[0].embedding.astype('float32') if faces else None

def load_dataset_embeddings():
    embeddings = []
    names = []
    for file in os.listdir(DATASET_DIR):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(DATASET_DIR, file)
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
    index = faiss.IndexFlatIP(EMBEDDING_DIM)  
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

if not os.path.exists(DATASET_DIR):
    st.error(f"Dataset directory '{DATASET_DIR}' not found. Please add your face images there.")
    st.stop()

with st.spinner("Loading face embeddings..."):
    embeddings, names = load_dataset_embeddings()
    if len(embeddings) == 0:
        st.error("No valid face embeddings found in dataset.")
        st.stop()
    index = build_faiss_index(embeddings)

st.success("Embeddings loaded.")

option = st.radio("Choose Input Method", ("📷 Webcam", "📁 Upload Image"))

if option == "📷 Webcam":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = app.get(rgb_frame)
        if faces:
            embedding = faces[0].embedding.astype('float32')
            name, dist = match_face(embedding, index, names)
            bbox = faces[0].bbox.astype(int)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({dist:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    cv2.destroyAllWindows()

elif option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
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