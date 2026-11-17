import torch.nn as nn
from .resnet import resnet18


class DilatedResNet18(nn.Module):
    def __init__(self, pretrained=True, use_dilation=True, use_maxpool=True):
        super(DilatedResNet18, self).__init__()
        self.use_dilation = use_dilation
        self.use_maxpool = use_maxpool

        # use_dilation: dilation instead of stride in layer3 and layer4
        replace_stride = [False, True, True] if self.use_dilation else [False, False, False]

        base_model = resnet18(pretrained=pretrained, replace_stride_with_dilation=replace_stride)

        # Everything except AvgPool and the FC layer
        layers = list(base_model.children())[:-2]

        if self.use_maxpool:
            # MVDet / SynMVCrowd baseline: keep MaxPool -> stride 8 with dilation (720x1280 -> 90x160)
            faithful_layers = layers
            freeze_limit = 6  # conv1, bn1, relu, maxpool, layer1, layer2
        else:
            # Legacy: drop MaxPool (index 3) -> stride 4 with dilation (720x1280 -> 180x320).
            # Only needed to load checkpoints trained before the switch to stride 8; the
            # feature_extractor indices differ, so a stride-4 state_dict fails loudly on a stride-8 model.
            faithful_layers = layers[:3] + layers[4:]
            freeze_limit = 5

        self.feature_extractor = nn.Sequential(*faithful_layers)
        self.stride = (8 if self.use_maxpool else 4) * (1 if self.use_dilation else 4)

        # conv1 .. layer2 always frozen
        for i in range(freeze_limit):
            for param in self.feature_extractor[i].parameters():
                param.requires_grad = False

        self.out_channels = 512

    def forward(self, x):
        is_5d = False
        if x.dim() == 5:
            is_5d = True
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)

        features = self.feature_extractor(x)

        if is_5d:
            _, C_out, H_out, W_out = features.shape
            features = features.view(B, N, C_out, H_out, W_out)

        return features
