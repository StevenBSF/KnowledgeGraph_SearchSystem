from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import time
import faiss
import numpy as np

# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'


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


# 加载CLIP模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# 初始化一个列表来保存图像的特征表示
image_features_list = []

# 记录检索开始的时间
start_time = time.time()

# 遍历图像路径列表，处理每个图像并提取特征
for image_path in image_files_path:
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt", padding=True)
    image_features = model.get_image_features(**image_inputs)
    image_features_list.append(image_features)

# 将图像特征列表堆叠成一个Tensor
image_features_stack = torch.stack(image_features_list)


def create_index(datas_embedding):
    # 构建索引
    index = faiss.IndexFlatIP(datas_embedding.shape[1])  # 这里必须传入一个向量的维度，创建一个空的索引
    index.add(datas_embedding)   # 把向量数据加入索引
    return index

def data_recall(faiss_index, query_embedding, top_k):
    Distance, Index = faiss_index.search(query_embedding, top_k)
    return Index

def data_recall_v2(faiss_index, query, top_k):
    query_embedding = model.encode([query])
    Distance, Index = faiss_index.search(query_embedding, top_k)
    return Index

def faiss_index_save(faiss_index, save_file_location):
    faiss.write_index(faiss_index, save_file_location)

def faiss_index_load(faiss_index_save_file_location):
    index = faiss.read_index(faiss_index_save_file_location)
    return index

# 假设image_features_stack是所有图像的特征向量堆叠成的Tensor
# 转换为NumPy数组，并确保数据类型为float32
feature_vectors = image_features_stack.detach().numpy()
feature_vectors = feature_vectors.squeeze()


save_file_location = "large.index"
faiss_index = faiss_index_load(save_file_location)
faiss_index.add(feature_vectors)
print(faiss_index.ntotal)

# 记录检索结束的时间
end_time = time.time()

# 计算并打印检索过程的总时间
print(f"Training took {end_time - start_time:.2f} seconds.")

