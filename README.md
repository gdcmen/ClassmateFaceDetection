# Real-Time Face Recognition System for Classroom Attendance

This project implements a real-time facial recognition system to automate classroom attendance. The system uses a webcam to detect and identify students’ faces, marking them as present if a match is found in the reference database. It is built using deep learning techniques, particularly leveraging **FaceNet embeddings** for robust recognition with limited training data.

## ✅ Working Approach: Embedding-Based Face Recognition

### 📌 Key Steps:
1. **Face Detection**: 
   - Uses **MTCNN (Multi-task Cascaded Convolutional Neural Networks)** to detect faces in real-time video frames.

2. **Embedding Extraction**:
   - Applies a **pretrained FaceNet model** (via `keras-facenet`) to convert each face into a 128-dimensional embedding vector.

3. **Similarity Matching**:
   - Compares the real-time face embedding with stored embeddings of enrolled students using **cosine similarity**.
   - If similarity exceeds a set threshold (e.g., 0.5), the face is considered recognized and attendance is marked.

4. **Attendance Report**:
   - Generates a summary showing which students were present during the session.

## 🧠 Pretrained Models Used

- **FaceNet** (via `keras-facenet`):  
  Used for embedding generation. Learns to map faces into a high-dimensional space where similar faces are close together.

- **MTCNN**:  
  Utilized for real-time face detection with high accuracy and robustness across varied lighting and angles.

## 🛠 Technologies & Libraries

- `TensorFlow/Keras`  
- `keras-facenet`  
- `mtcnn`  
- `OpenCV`  
- `NumPy`

## 💡 Highlights

- Works with as little as **one photo per student**.
- **No training required** thanks to transfer learning with FaceNet.
- Fast and lightweight: runs in real time on a regular laptop.
- Easily extendable: new students can be added by inserting their face embeddings into the database.

---

