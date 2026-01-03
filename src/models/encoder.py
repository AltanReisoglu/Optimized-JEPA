import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import math
import dataclasses
from abc import ABC, abstractmethod

from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclasses.dataclass
class VJEP2EncoderConfig:
    processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')
    vjepa2_vit_large = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')
    VIDEO_INPUT_SIZE = (224, 224)
    VIDEO_INPUT_FRAMES = 16

    predictor_id = "Qwen/Qwen2.5-1.5B"

    tokenizer_predictor = AutoTokenizer.from_pretrained(predictor_id)
    model_predictor = AutoModelForCausalLM.from_pretrained(
        predictor_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    from sentence_transformers import SentenceTransformer

    auxiliary_model = SentenceTransformer("all-MiniLM-L6-v2")

class WorldModel(ABC):
    @abstractmethod
    def encode(self, video, text):
        pass
    
    @abstractmethod
    def decode(self, video, text):
        pass

    @abstractmethod
    def reward(self, video, text):
        pass

class VJEP2Encoder(WorldModel,nn.Module):
    def __init__(self, config: VJEP2EncoderConfig):
        super().__init__()
        self.config = config
        self.processor = config.processor
        self.encoder = config.vjepa2_vit_large
        for layer in self.encoder.parameters():
            layer.requires_grad = False

        self.auxiliary_model = config.auxiliary_model
        self.proj=nn.Linear(300,768)

    def encode(self, path):
        video = self.processor(path, return_tensors="pt").to("cuda")
        video = self.encoder(video)
        return video
    def embed_text(self, text):
        text = self.auxiliary_model.encode(text)
        return text
    def decode(self, video, text):
        pass
    
    def forward(self, path,text):
        print(self.encode(self.embed_text(text)).shape,self.embed_text(text).shape)


if __name__ == "__main__":
    config = VJEP2EncoderConfig()
    encoder = VJEP2Encoder(config)
    print(encoder(r"C:\Users\bahaa\Desktop\Optimized-JEPA\assets\vecteezy_a-cyclist-rides-on-a-bike-path-in-the-city_27944714.mov","text"))
    
    
