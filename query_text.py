from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import time
import faiss
import numpy as np

# 定义根文件夹路径
img_folder_path = './photo'

# 初始化一个列表来保存所有图像的路径
image_files_path = []

# 遍历根文件夹下的每个数字命名的子文件夹
for folder_name in os.listdir(img_folder_path):
    folder_path = os.path.join(img_folder_path, folder_name)
    if os.path.isdir(folder_path):
        # 遍历子文件夹中的所有文件
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.jpg'):
                # 如果是.jpg文件，将其路径添加到列表中
                image_files_path.append(os.path.join(folder_path, file_name))


def data_recall(faiss_index, query_embedding, top_k):
    Distance, Index = faiss_index.search(query_embedding, top_k)
    return Index

def faiss_index_save(faiss_index, save_file_location):
    faiss.write_index(faiss_index, save_file_location)

def faiss_index_load(faiss_index_save_file_location):
    index = faiss.read_index(faiss_index_save_file_location)
    return index


# 加载CLIP模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


query_text = "function"

# 处理文本输入
text_inputs = processor(text=query_text, return_tensors="pt", padding=True, truncation=True)

# 使用CLIP模型提取文本特征
text_features = model.get_text_features(**text_inputs)

query_vector = text_features.detach().numpy()

save_file_location = "large.index"
faiss_index = faiss_index_load(save_file_location)

# 搜索最相似的k个图像
k = 10  # 例如，返回最相似的10个图像
sim_data_Index = data_recall(faiss_index, query_vector, k)

# 输出最相似的图像路径
print("Most similar images:")
for idx in sim_data_Index[0]:  # I[0]包含了最相似图像的索引
    print(image_files_path[idx])
