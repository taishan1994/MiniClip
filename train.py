import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import json
import random
import faiss
import torch
import numpy as np
import torch.nn.functional as F

from model import MiniClip
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer
from transform import image_transform
from tokenizer import tokenize


# 构建dataset
class MiniDataset(Dataset):
    def __init__(self, train_path, image_size, is_train=True):
        with open(train_path, "r") as fp:
            self.data = json.load(fp)
        self.root = "/data/gongoubo/MiniClip/data"
        self.tokenizer = tokenize
        self.process = image_transform(image_size, is_train=is_train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        d = self.data[item]
        image = d["image"]
        texts = d["caption"]
        text = [random.choice(texts)]

        image = image.replace("\\", "/")
        image = os.path.join(self.root, image)
        image = Image.open(image).convert("RGB")

        image_input = self.process(image)
        text_input = self.tokenizer(text).squeeze(0)

        out = {
            "text": text_input,
            "image": image_input
        }

        return out


#
cfg_path = "model_configs/TinyCLIP-ViT-40M-32-Text-19M.json"
clip = MiniClip(cfg_path)

for k, v in clip.named_parameters():
    print(k, v.shape)

state_dict = torch.load("/data/gongoubo/MiniClip/model_hub/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M.pt",
                        map_location="cpu")
new_state_dict = {}
for k, v in state_dict["state_dict"].items():
    if "visual" in k:
        new_state_dict[k.replace("module", "image_encoder")] = v
    elif "logit_scale" in k:
        new_state_dict[k.replace("module", "logit_scale")] = v
    else:
        new_state_dict[k.replace("module", "text_encoder")] = v

# clip.load_state_dict(new_state_dict, strict=True)

num_train_epochs = 2000
train_batch_size = 16

# 构建dataloader
train_path = "data/en_val.json"
train_dataset = MiniDataset(train_path, clip.image_encoder.visual.image_size)

# train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=8, shuffle=True)

training_args = TrainingArguments(
    output_dir='./checkpoints',  # output directory 结果输出地址
    num_train_epochs=num_train_epochs,  # total # of training epochs 训练总批次
    per_device_train_batch_size=train_batch_size,  # batch size per device during training 训练批大小
    logging_dir='./logs/',  # directory for storing logs 日志存储位置
    learning_rate=3e-5,  # 学习率
    save_steps=False,  # 不保存检查点
    logging_strategy="steps",
    logging_steps=1,
    max_grad_norm=1,
    do_eval=False,
    do_train=True,
)


class MiniTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        image_features, text_features, logit_scale = outputs
        logits_per_image = image_features @ text_features.T
        logits_per_text = text_features @ image_features.T
        num_logits = logits_per_image.shape[0]
        device = image_features.device
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_text, labels)
                     ) / 2

        return total_loss


trainer = MiniTrainer(
    model=clip,  # the instantiated 🤗 Transformers model to be trained 需要训练的模型
    args=training_args,  # training arguments, defined above 训练参数
    train_dataset=train_dataset,  # training dataset 训练集
)

trainer.train()
trainer.save_model()
