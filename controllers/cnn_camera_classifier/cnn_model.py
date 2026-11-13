"""CNN for CIFAR-10 (32 x 32 RGB images, 10 classes).

The same file is used by the training notebook and by the Webots controller,
so the network that is trained is exactly the network that runs on the robot.
"""
import torch
import torch.nn as nn

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

# Per-channel mean and standard deviation of the CIFAR-10 training set.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def conv_block(c_in, c_out, dropout):
    """Two 3x3 convolutions (each followed by batch norm and ReLU), then 2x2 max pooling."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),          # halves the height and width
        nn.Dropout(dropout),
    )


class CifarCNN(nn.Module):
    """VGG-style network: three convolution blocks and a two-layer classifier.

    Input 3 x 32 x 32 -> block 1 (32 ch, 16 x 16) -> block 2 (64 ch, 8 x 8)
    -> block 3 (128 ch, 4 x 4) -> fully connected 256 -> 10 class scores.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32, 0.2),
            conv_block(32, 64, 0.3),
            conv_block(64, 128, 0.4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def preprocess(rgb_uint8):
    """Turn an H x W x 3 uint8 RGB tensor into a normalised 1 x 3 x 32 x 32 batch.

    The largest centred square is cropped first, so the object is not squashed,
    and then resized to 32 x 32 with antialiasing.
    """
    img = rgb_uint8.permute(2, 0, 1).float().div(255.0)          # 3 x H x W, range 0..1
    _, h, w = img.shape
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    img = img[:, top:top + s, left:left + s]
    img = nn.functional.interpolate(img.unsqueeze(0), size=(32, 32), mode="bilinear",
                                    align_corners=False, antialias=True)
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    return (img - mean) / std
