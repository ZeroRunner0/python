import torch
from torch import nn
from d2l import torch as d2l
from torchsummary import summary
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter


class DanLu():
    def __init__(self, net, train_dataset, val_dataset, name="model"):
        self.net_name = name
        self.net = net
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.accuracy_rate = 0

    def train(self, epochs):
        print(f"training on {self.device}")

        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=self.net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.1,         # 每次减少学习率的倍数
            patience=5,         # 在多少个epoch内监控指标不改善后再进行学习率衰减
            min_lr=0.00001,     # 学习率的下限
        )

        self.net.to(self.device)
        for epoch in range(epochs):
            # 将网络设置成训练模式
            self.net.train()
            # 可视化训练进度条
            train_bar = tqdm(self.train_dataset)
            # 计算每个epoch的loss总和
            loss_sum = 0.0
            for i, (train_img, train_label) in enumerate(train_bar):
                optimizer.zero_grad()

                train_img, train_label = train_img.to(self.device), train_label.to(self.device)
                output = self.net(train_img)
                loss = loss_func(output, train_label)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                # tqdm增加前缀
                train_bar.desc = f'train epoch:[{epoch + 1}/{epochs}], loss:{loss:.5f}'

            self.validate()
            # 正确率不再增加时，降低学习率
            lr_scheduler.step(loss_sum)
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f'current lr:{current_lr},'
                f'train loss:{(loss_sum / len(self.train_dataset)):.5f}, '
                f'val correct rate:{self.accuracy_rate:.5f}')

    def validate(self):
        if isinstance(self.net, nn.Module):
            # NET变成验证模式
            self.net.eval()

        correct_num = 0
        with torch.no_grad():
            for i, (val_img, val_label) in enumerate(self.val_dataset):
                val_img, val_label = val_img.to(self.device), val_label.to(self.device)
                output = self.net(val_img)
                # index表示类别
                max_val, index = torch.max(output, dim=1)
                correct_num += torch.sum(index == val_label).item()

            accuracy_rate = correct_num / len(self.val_dataset)
            # 更新模型的预测正确率
            if accuracy_rate > self.accuracy_rate:
                self.accuracy_rate = accuracy_rate
                torch.save(self.net.state_dict(), str(self.net_name) + ".pth")






class NIN(nn.Module):
    """
    继承nn.Module类
    """
    def __init__(self, num_of_label):
        """
        类实例化时自动调用
        """
        super(NIN, self).__init__()
        self.net = nn.Sequential(
            self.nin_block(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0), nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2), nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1), nn.Dropout(0.5),
            self.nin_block(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1), nn.Dropout(0.5),
            self.nin_block(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1), nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(in_channels=256, out_channels=num_of_label, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.init_weight()

    def forward(self, x):
        return self.net(x)

    def nin_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding), nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0), nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0), nn.ReLU(),
        )

    def init_weight(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, val=0)

    def summary_net_shape(self):
        summary(self.net, (3, 227, 227), device="cpu")


# 定义数据变换（假设有一个自定义的数据变换函数 data_transform）
def data_transform(phase):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif phase == 'val':
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError("Unknown phase. Use 'train' or 'val'.")


dataset_path = "E:/personal/workspace/python/dataset/"
train_dataset = torchvision.datasets.CIFAR10(root=dataset_path,
                                             train=True,
                                             download=True,
                                             transform=data_transform("train"))
val_dataset = torchvision.datasets.CIFAR10(root=dataset_path,
                                           train=False,
                                           download=True,
                                           transform=data_transform("val"))
# 创建子集
subset_indices = list(range(0, len(train_dataset), 10))
train_subset = Subset(train_dataset, subset_indices)

subset_indices = list(range(0, len(val_dataset), 10))
val_subset = Subset(val_dataset, subset_indices)


train_dataset_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
val_dataset_loader = DataLoader(val_subset, batch_size=1)

print(f'{len(train_dataset_loader)} for training, {len(val_dataset_loader)} for validation')

nin_net = NIN(10)

# for o, (test_img, test_label) in enumerate(train_dataset_loader):
#     out = nin_net.net(test_img)
#     print(out, out.shape)
#     print(o, test_label, test_label.shape)
#     loss = nn.CrossEntropyLoss(out, test_label)
#     loss.backward()
#     print(loss.item())

nin_DaLu = DanLu(nin_net.net, train_dataset_loader, val_dataset_loader, name="NIN")
nin_DaLu.train(50)


print("code end")

