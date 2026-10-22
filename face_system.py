import cv2
import torch
import numpy as np

from facenet_pytorch import MTCNN, InceptionResnetV1

device = "cuda" if torch.cuda.is_available() else "cpu"

face = MTCNN(image_size=160,margin=20,keep_all=False,device=device)

model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def get_embedding(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    face_tensor = face(image)
    if face_tensor is None:
        return None
    face_tensor = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(face_tensor)
        embedding = torch.nn.functional.normalize(embedding,p=2,dim=1)
    return embedding.cpu().numpy()[0]

def register_face(image_path, save_path="my_face.npy"):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    embedding = get_embedding(image)
    if embedding is None:
        print("No face detected")
        return False
    np.save(save_path, embedding)
    print("Face registered successfully")
    return True

def compare_face(image, reference_path="my_face.npy"):
    try:
        reference = np.load(reference_path)
    except FileNotFoundError:
        print("Register your face first")
        return None
    embedding = get_embedding(image)
    if embedding is None:
        return None
    similarity = float(np.dot(reference, embedding))
    return similarity
