import gradio as gr
import os
import json
import faiss
import numpy as np

from model import MiniClip
from PIL import Image
from transform import image_transform
from tokenizer import tokenize
from tqdm import tqdm
from safetensors.torch import load_file


cfg_path = "model_configs/TinyCLIP-ViT-40M-32-Text-19M.json"
state_dict_path = "/data/gongoubo/MiniClip/checkpoints/model.safetensors"
clip = MiniClip(cfg_path)

state_dict = load_file(state_dict_path)
for k,v in state_dict.items():
    print(k, v.shape)
clip.load_state_dict(state_dict, strict=True)

image_features = np.load("output/image2.npy").astype('float32')
d = image_features.shape[1]
index = faiss.IndexFlatL2(d)
index.add(image_features)
with open("data/en_val.json", "r") as fp:
    data = json.loads(fp.read())
image_paths = {i:os.path.join("/data/gongoubo/MiniClip/data", d["image"].replace("\\", "/")) for i,d in enumerate(data)}

# 处理文本 query -> 特征向量
def encode_text(query):
    text_input = tokenize(query)
    text_features = clip.encode_text(text_input, normalized=True)
    text_features = text_features.detach().cpu().numpy().astype('float32')
    return text_features


# 检索函数
def search_images(query, top_k=20):
    text_vector = encode_text(query)  # 确保数据类型匹配 FAISS
    print(text_vector.shape)
    _, indices = index.search(text_vector, top_k)  # 检索 top_k 个最相似图片
    retrieved_images = [Image.open(image_paths[i]) for i in indices[0]]  # 加载图片
    return retrieved_images


# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 文本检索图片")
    with gr.Row():
        query_input = gr.Textbox(label="输入查询文本")
        search_button = gr.Button("搜索")

    gallery = gr.Gallery(label="检索结果", columns=[10], height=300)  # 以网格展示图片

    search_button.click(fn=search_images, inputs=[query_input], outputs=[gallery])

# 运行 Gradio
demo.launch(server_name="0.0.0.0", server_port=7860)
