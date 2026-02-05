import torch
import torch.nn as nn




def get_model(param):
    if param["model"] == "default":
        return DefaultModel(param["num_classes"])







class DefaultModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), padding=1),        # 32 x 32 x 32
            nn.ELU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, (3, 3), padding=1),       # 32 x 32 x 32
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2)),           # 32 x 16 x 16
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, (3, 3), padding=1),       # 64 x 16 x 16
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), padding=1),       # 64 x 16 x 16
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2)),           # 64 x 8 x 8
            nn.Dropout2d(0.3),

            nn.Conv2d(64, 128, (3, 3), padding=1),      # 128 x 8 x 8
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), padding=1),     # 128 x 8 x 8
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2)),            # 128 x 4 x 4
            nn.Dropout2d(0.4),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x_train):
        return self.main(x_train)



def weight_init(layer):  #初始化权重
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
        layer.bias.data.zero_()
    # elif isinstance(m, nn.BatchNorm3d):
    #     m.weight.data.fill_(1)
    #     m.bias.data.zero_()
    # elif isinstance(m, nn.Linear):
    #     m.weight.data.normal_(0, 0.02)
    #     m.bias.data.zero_()
