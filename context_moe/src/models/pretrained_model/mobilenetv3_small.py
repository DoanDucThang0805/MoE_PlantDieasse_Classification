# import timm
# from torchinfo import summary


# num_classes=8
# # Load model pretrained
# model = timm.create_model(
#     'mobilenetv3_small_100.lamb_in1k',
#     pretrained=True,
#     num_classes=num_classes
# )


# # Summary
# summary(model, (1,3,224,224), col_names=["input_size", "output_size", "num_params", "mult_adds"])


from torch import nn
from torchinfo import summary
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


NUM_CLASSES = 8

model = mobilenet_v3_small(
    weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
)
model.classifier[-1] = nn.Linear(
    model.classifier[-1].in_features,
    NUM_CLASSES,
)


if __name__ == "__main__":
    summary(
        model,
        (1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
    )

