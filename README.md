# final-project-ai-facial-regogniton
# Facial recognition project code for CAP 4630
# AI Facial Recognition for Security & Law Enforcement
#Matthew IshibashiHeron 

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Load faces: only identities with at least 50 images to keep it manageable
lfw = fetch_lfw_people(min_faces_per_person=50, resize=0.5, color=False)

X_images = lfw.images          # shape: (n_samples, h, w)
y = lfw.target                 # integer labels
target_names = lfw.target_names
n_samples, h, w = X_images.shape

print("Number of samples:", n_samples)
print("Image shape:", (h, w))
print("Number of identities (classes):", len(target_names))
print("Identities:", target_names)

# Show a grid of sample faces to put into your slides
def plot_sample_faces(images, labels, target_names, n_rows=2, n_cols=5):
    plt.figure(figsize=(10, 4))
    for i in range(n_rows * n_cols):
        if i >= len(images):
            break
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(target_names[labels[i]][:15], fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

plot_sample_faces(X_images, y, target_names)

# Flatten images (h, w) → (h*w) feature vectors
X = X_images.reshape((n_samples, -1))
print("Feature matrix shape:", X.shape)

# Train/Test split (stratified = keeps class proportions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# Use PCA to reduce dimensionality (feature extraction)
n_components = 150  # you can tune this (hyperparameter)

pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True, random_state=42)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("Original dim:", X_train.shape[1])
print("Reduced dim:", X_train_pca.shape[1])

# Train an SVM classifier on PCA embeddings (identity recognition)
svm_clf = SVC(
    kernel="rbf",
    class_weight="balanced",
    C=10.0,
    gamma=0.001,
    probability=False,
    random_state=42
)

svm_clf.fit(X_train_pca, y_train)

# Predict on test set
y_pred = svm_clf.predict(X_test_pca)

# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Macro-averaged precision, recall, F1
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro", zero_division=0
)

print("Precision (macro):", precision)
print("Recall (macro):", recall)
print("F1-score (macro):", f1)

# Detailed per-class report (you can show a snippet in your report, not all in slides)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Identity Recognition)")
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45, ha="right", fontsize=6)
plt.yticks(tick_marks, target_names, fontsize=6)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# Define a simulated police watchlist: first 3 identities
watchlist_ids = list(range(min(3, len(target_names))))
watchlist_names = target_names[watchlist_ids]
print("Watchlist identities:", watchlist_names)

# Create binary labels: 1 = on watchlist, 0 = not on watchlist
y_train_watchlist = np.isin(y_train, watchlist_ids).astype(int)
y_test_watchlist = np.isin(y_test, watchlist_ids).astype(int)

# Train SVM for watchlist vs non-watchlist
watchlist_clf = SVC(
    kernel="rbf",
    class_weight="balanced",
    C=10.0,
    gamma=0.001,
    probability=True,
    random_state=42
)

watchlist_clf.fit(X_train_pca, y_train_watchlist)

y_pred_watchlist = watchlist_clf.predict(X_test_pca)

# Evaluate
acc_w = accuracy_score(y_test_watchlist, y_pred_watchlist)
prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
    y_test_watchlist, y_pred_watchlist, average="binary", zero_division=0
)

print("\nWatchlist vs Non-Watchlist Classification:")
print("Accuracy:", acc_w)
print("Precision (watchlist=1):", prec_w)
print("Recall (watchlist=1):", rec_w)
print("F1 (watchlist=1):", f1_w)

cm_w = confusion_matrix(y_test_watchlist, y_pred_watchlist)
print("Confusion Matrix (rows=true, cols=pred):\n", cm_w)

import os
import cv2
import face_recognition
import numpy as np

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.5
FRAME_RESIZE_SCALE = 0.25
MODEL = "hog"

# Load known faces
known_face_encodings = []
known_face_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if not os.path.isdir(person_dir):
        continue

    for filename in os.listdir(person_dir):
        filepath = os.path.join(person_dir, filename)
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            continue

        known_face_encodings.append(encodings[0])
        known_face_names.append(name)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, TOLERANCE)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        name = "Unknown"
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top = int(top / FRAME_RESIZE_SCALE)
        right = int(right / FRAME_RESIZE_SCALE)
        bottom = int(bottom / FRAME_RESIZE_SCALE)
        left = int(left / FRAME_RESIZE_SCALE)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom + 20), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 2, bottom + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("AI Facial Recognition Demo (Press Q to Quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
