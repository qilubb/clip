import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
import torch
import clip

data_path = "D:/clip/ChinaSet/ChinaSet_AllFiles"
image_folder = os.path.join(data_path, "CXR_png")
text_folder = os.path.join(data_path, "ClinicalReadings")

desired_width = 224
desired_height = 224

# 加载图像和文本特征
def load_data():
    images = []
    texts = []

    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert("RGB")
            image = image.resize((desired_width, desired_height))
            images.append(np.array(image))

            text_filename = filename.replace(".png", ".txt")
            text_path = os.path.join(text_folder, text_filename)
            with open(text_path, "r") as file:
                text = file.read().strip()
                texts.append(text)

    images = np.array(images)
    texts = np.array(texts)

    return images, texts

# 加载数据
images, texts = load_data()

# 打印图像特征的形状
print("Image features shape:", images.shape)

# 加载CLIP模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)
model.to(device)

# 提取图像特征
image_features = []
for i in range(len(images)):
    image = images[i]
    image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model.encode_image(image)
        print("image_feature shape",image_feature.shape)
    image_feature = image_feature.flatten().cpu().numpy()
    image_features.append(image_feature)

image_features = np.array(image_features)
print("image_features shape",image_features.shape)

# 提取文本特征
text_features = []
for i in range(len(texts)):
    text = texts[i]
    print("text", text)
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        # print("text",text)
        text_feature = model.encode_text(text)
        print("text_feature shape", text_feature.shape)
    text_feature = text_feature.flatten().cpu().numpy()
    text_features.append(text_feature)

text_features = np.array(text_features)
print("text_features shape",text_features.shape)

# 对图像特征和文本特征执行t-SNE
tsne = TSNE(n_components=2, random_state=42)
image_reduced = tsne.fit_transform(image_features)
text_reduced = tsne.fit_transform(text_features)

# 可视化t-SNE降维结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(image_reduced[:, 0], image_reduced[:, 1], c="blue")
plt.title("Image Features")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.scatter(text_reduced[:, 0], text_reduced[:, 1], c="red")
plt.title("Text Features")
plt.axis('off')
plt.tight_layout()
plt.show()

# 对图像特征和文本特征执行t-SNE
tsne = TSNE(n_components=2, random_state=42)
combined_features = np.concatenate((image_features, text_features), axis=1)
reduced_combined = tsne.fit_transform(combined_features)
print("combined_features.shape",combined_features.shape)

# 可视化t-SNE降维结果
plt.figure(figsize=(10, 5))
# plt.scatter(reduced_combined[:len(image_features), 0], reduced_combined[:len(image_features), 1], c="blue", label="Image Features")
# plt.scatter(reduced_combined[len(image_features):, 0], reduced_combined[len(image_features):, 1], c="red", label="Text Features")
plt.scatter(reduced_combined[:100, 0], reduced_combined[:100, 1], c=range(100), label="combined_features")
plt.title("Combined Features")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.show()