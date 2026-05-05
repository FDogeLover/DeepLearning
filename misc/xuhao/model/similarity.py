import clip
import torch
from PIL import Image
import os

class CLIPSimilarity:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)


    def get_image_embedding(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features / image_features.norm(dim=-1, keepdim=True)

    def get_text_embedding(self, text):
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def image_to_image_similarity(self, img_path1, img_path2):
        feat1 = self.get_image_embedding(img_path1)
        feat2 = self.get_image_embedding(img_path2)
        similarity = (feat1 @ feat2.T).item()
        return round(similarity, 4)

    def image_to_prompt_similarity(self, img_path, prompt_text):
        img_feat = self.get_image_embedding(img_path)
        text_feat = self.get_text_embedding(prompt_text)
        similarity = (img_feat @ text_feat.T).item()
        return round(similarity, 4)
