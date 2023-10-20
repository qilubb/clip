import torch
import clip
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)

image_path = "CXR1021_IM-0017-1001-0001.png"
text_input = clip.tokenize(["Recurrent right pneumothorax, complete collapse of the right lung, near 100%. Right-to-left mediastinal shift is present, suggesting XXXX physiology."]).to(device)

def extract_image_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features.cpu().numpy()

def extract_text_features(text_input):
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    return text_features.cpu().numpy()

def visualize_tsne(embeddings, labels, title):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings[:, 0], embeddings[:, 1])
    for i, label in enumerate(labels):
        x, y = embeddings[i]
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    plt.axis("off")
    plt.title(title)
    plt.show()

image_feature = extract_image_features(image_path)
text_feature = extract_text_features(text_input)

embeddings = np.concatenate([image_feature, text_feature], axis=0)
labels = ["image", "text"]
visualize_tsne(embeddings, labels, "Image and Text Embeddings")