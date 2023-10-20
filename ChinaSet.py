import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer

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
            image = Image.open(image_path).convert("L")
            image = image.resize((desired_width, desired_height))
            images.append(np.array(image))

            text_filename = filename.replace(".png", ".txt")
            text_path = os.path.join(text_folder, text_filename)
            with open(text_path, "r") as file:
                text = file.read().strip()
                texts.append(text)

    return np.array(images), np.array(texts)

# 加载数据
images, texts = load_data()

# 打印图像特征的形状
print("Image features shape:", images.shape)

# 对图像特征执行t-SNE
tsne_image = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
image_reduced = tsne_image.fit_transform(images.reshape(images.shape[0], -1))

# 对文本特征进行向量化
vectorizer = CountVectorizer()
text_features = vectorizer.fit_transform(texts)

# 对文本特征执行t-SNE
tsne_text = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
text_reduced = tsne_text.fit_transform(text_features.toarray())

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