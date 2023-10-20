import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import torch
import clip

data_path = "D:/clip/ChinaSet/ChinaSet_AllFiles"
image_folder = os.path.join(data_path, "CXR_png")
text_folder = os.path.join(data_path, "ClinicalReadings")
folder_path = os.path.join(data_path, "ClinicalReadings")

desired_width = 224
desired_height = 224

labels = []  # 标签列表

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):  # 只处理以 .txt 结尾的文本文件
        file_path = os.path.join(folder_path, filename)  # 构建文件完整路径
        with open(file_path, "r") as file:
            text = file.read()  # 读取文本内容

            # 根据文本内容是否包含关键词 "normal" 打上标签
            if "normal" in text:
                labels.append(0)
            else:
                labels.append(1)

labels = np.array(labels)  # 转换为 NumPy 数组


# 加载文本特征
def load_data():
    images = []
    texts = []

    for filename in os.listdir(text_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(text_folder, filename)
            with open(file_path, "r") as file:
                text = file.readline().strip()
                texts.append(text)

    texts = np.array(texts)

    return texts

# 加载数据
texts = load_data()

# 加载CLIP模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)
model.to(device)


# 提取文本特征
text_features = []
for i in range(len(texts)):
    text = texts[i]
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        # print("text",text)
        text_feature = model.encode_text(text)
    text_feature = text_feature.flatten().cpu().numpy()
    text_features.append(text_feature)

text_features = np.array(text_features)


# 将数据集分成训练集和测试集
train_images, test_images, train_labels, test_labels = train_test_split(text_features, labels, test_size=0.1, random_state=42)


# 创建SVM和MLP分类器
svm = SVC(kernel='linear', C=1, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# 训练SVM和MLP分类器
svm.fit(train_images, train_labels)
mlp.fit(train_images, train_labels)

# 使用训练数据对模型进行训练
svm_pred = svm.predict(test_images)
mlp_pred = mlp.predict(test_images)

# 计算模型的准确率
svm_acc = accuracy_score(test_labels, svm_pred)
mlp_acc = accuracy_score(test_labels, mlp_pred)

print("SVM准确率：", svm_acc)
print("MLP准确率：", mlp_acc)
