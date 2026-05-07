import torch
import torch.nn as nn
import timm


def build_model(device):
    model = timm.create_model("efficientnet_b2", pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1408, 4),
    )
    model = model.to(device)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    print(model.classifier)

    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    output = model(dummy_input)
    print(output.shape)
