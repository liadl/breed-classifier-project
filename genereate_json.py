import json
from torchvision import datasets

train_dataset = datasets.OxfordIIITPet(root="./dataset", split="trainval", download=False)

with open("class_names.json", "w") as f:
    json.dump(train_dataset.classes, f)