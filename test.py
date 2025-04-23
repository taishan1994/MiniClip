import json
import os

import faiss
import torch
import numpy as np

from model import MiniClip
from PIL import Image
from transform import image_transform
from tokenizer import tokenize
from tqdm import tqdm


def load_ori_model(cfg_path, state_dict_path):
    clip = MiniClip(cfg_path)

    for k, v in clip.named_parameters():
        print(k, v.shape)

    state_dict = torch.load(state_dict_path, map_location="cpu")
    new_state_dict = {}
    for k, v in state_dict["state_dict"].items():
        if "visual" in k:
            new_state_dict[k.replace("module", "image_encoder")] = v
        elif "logit_scale" in k:
            new_state_dict[k.replace("module", "logit_scale")] = v
        else:
            new_state_dict[k.replace("module", "text_encoder")] = v

    clip.load_state_dict(new_state_dict, strict=True)
    return clip


def predict_one_sample(model):
    img_path = "data/dog.png"
    text = ["a dog", "a cat", "a fish", "a pig"]

    image = Image.open(img_path).convert("RGB")
    val_processor = image_transform(model.image_encoder.visual.image_size, is_train=False)

    image_input = val_processor(image).unsqueeze(0)
    text_input = tokenize(text)

    img_feature = model.encode_image(image_input, normalized=True)
    text_feature = model.encode_text(text_input, normalized=True)

    img_feature = img_feature.detach().cpu().numpy()
    text_feature = text_feature.detach().cpu().numpy()
    print(text_feature @ img_feature.T)


def test_on_flickr(model):
    root = "/data/gongoubo/MiniClip/data"
    with open("data/en_val.json", "r") as fp:
        data = json.loads(fp.read())

    text_features = []
    image_features = []
    for i, d in tqdm(enumerate(data), total=len(data)):
        caption = d["caption"]
        image = d["image"].replace("\\", "/")
        image = os.path.join(root, image)
        # 取第0个caption
        caption = caption[:1]
        image = Image.open(image).convert("RGB")
        val_processor = image_transform(model.image_encoder.visual.image_size, is_train=False)

        image_input = val_processor(image).unsqueeze(0)
        text_input = tokenize(caption)

        img_feature = model.encode_image(image_input, normalized=True)
        text_feature = model.encode_text(text_input, normalized=True)

        img_feature = img_feature.detach().cpu().numpy()
        text_feature = text_feature.detach().cpu().numpy()
        text_features.append(text_feature[0])
        image_features.append(img_feature[0])

    text_features = np.stack(text_features, axis=0)
    image_features = np.stack(image_features, axis=0)

    np.save("output/text2.npy", text_features)
    np.save("output/image2.npy", image_features)


def search_by_faiss():
    text_features = np.load("output/text2.npy").astype('float32')
    image_features = np.load("output/image2.npy").astype('float32')
    d = text_features.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(image_features)
    top1 = 0
    top3 = 0
    top5 = 0
    top10 = 0

    with open("data/en_val.json", "r") as fp:
        data = json.loads(fp.read())
    id2query = {i:d["caption"] for i,d in enumerate(data)}

    for i, text_feature in enumerate(text_features):
        distances, indices = index.search(np.array([text_feature]), k=10)
        # print(indices)
        inds = indices[0].tolist()
        if i == inds[0]:
            top1 += 1
            print(id2query[i])
        if i in inds[:3]:
            top3 += 1
        if i in inds[:5]:
            top5 += 1
        if i in inds[:10]:
            top10 += 1

    print("top1  acc：", top1 / 1000 * 100)
    print("top3  acc：", top3 / 1000 * 100)
    print("top5  acc：", top5 / 1000 * 100)
    print("top10 acc：", top10 / 1000 * 100)


def load_trained_model(cfg_path, state_dict_path):
    from safetensors.torch import load_file
    clip = MiniClip(cfg_path)
    # for k, v in clip.named_parameters():
    #     print(k, v.shape)
    #
    state_dict = load_file(state_dict_path)
    for k,v in state_dict.items():
        print(k, v.shape)
    clip.load_state_dict(state_dict, strict=True)
    return clip


if __name__ == '__main__':
    # cfg_path = "model_configs/TinyCLIP-ViT-40M-32-Text-19M.json"
    # state_dict_path = "/data/gongoubo/MiniClip/model_hub/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M.pt"
    # clip = load_ori_model(cfg_path, state_dict_path)
    # # clip = MiniClip(cfg_path)
    # predict_one_sample(clip)
    # test_on_flickr(clip)
    # search_by_faiss()

    cfg_path = "model_configs/TinyCLIP-ViT-40M-32-Text-19M.json"
    state_dict_path = "/data/gongoubo/MiniClip/checkpoints/model.safetensors"
    clip = load_trained_model(cfg_path, state_dict_path)

    # predict_one_sample(clip)
    test_on_flickr(clip)
    search_by_faiss()
