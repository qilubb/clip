import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载预训练好的模型
model, preprocess = clip.load("ViT-B/32", device=device)

# 读取艾斯的图片和候选类别文字
image = preprocess(Image.open("CXR1021_IM-0017-1001-0001.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a man", "a dog", "a cat"]).to(device)

with torch.no_grad():
    # 计算每一张图像和每一个文本的相似度值
    logits_per_image, logits_per_text = model(image, text)

    # 对该image与每一个text的相似度值进行softmax
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)