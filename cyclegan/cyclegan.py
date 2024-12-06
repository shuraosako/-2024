import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np

# Generatorネットワーク
class Generator(nn.Module):
    def __init__(self, input_channels=3):
        super(Generator, self).__init__()

        # 初期の畳み込み層
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # ダウンサンプリング部分
        self.down_blocks = nn.Sequential(
            self._downsample(64, 128),
            self._downsample(128, 256)
        )

        # レジデュアルブロック
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(9)]
        )

        # アップサンプリング部分
        self.up_blocks = nn.Sequential(
            self._upsample(256, 128),
            self._upsample(128, 64)
        )

        # 出力層
        self.output = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )

    def _downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _upsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down_blocks(x)
        x = self.residual_blocks(x)
        x = self.up_blocks(x)
        return self.output(x)

# レジデュアルブロック
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

# Discriminatorネットワーク
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            self._discriminator_block(input_channels, 64, normalize=False),
            self._discriminator_block(64, 128),
            self._discriminator_block(128, 256),
            self._discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def _discriminator_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# カスタムデータセット
class StatueHumanDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.statue_dir = os.path.join(root_dir, 'statues')
        self.human_dir = os.path.join(root_dir, 'humans')
        self.statue_images = sorted(os.listdir(self.statue_dir))
        self.human_images = sorted(os.listdir(self.human_dir))

    def __len__(self):
        return max(len(self.statue_images), len(self.human_images))

    def __getitem__(self, idx):
        statue_idx = idx % len(self.statue_images)
        human_idx = idx % len(self.human_images)

        statue_path = os.path.join(self.statue_dir, self.statue_images[statue_idx])
        human_path = os.path.join(self.human_dir, self.human_images[human_idx])

        statue_img = Image.open(statue_path).convert('RGB')
        human_img = Image.open(human_path).convert('RGB')

        if self.transform:
            statue_img = self.transform(statue_img)
            human_img = self.transform(human_img)

        return {'A': statue_img, 'B': human_img}

class Config:
    def __init__(self):
        # self.epoch = 0
        self.epoch = 0
        # self.n_epochs = 200
        self.n_epochs = 5000
        self.batch_size = 1
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.decay_epoch = 100
        self.img_size = 256
        self.channels = 3
        self.lambda_cycle = 10.0
        self.lambda_identity = 5.0
        self.model_save_path = "/content/drive/MyDrive/チャレキャラ/model6_241206"

def save_models(G_AB, G_BA, D_A, D_B, epoch, save_path):
    """モデルを保存する関数"""
    save_path = os.path.join(save_path, "model")
    os.makedirs(save_path, exist_ok=True)

    torch.save(G_AB.state_dict(), os.path.join(save_path, f'model_generator_AB_{epoch}.pth'))
    torch.save(G_BA.state_dict(), os.path.join(save_path, f'model_generator_BA_{epoch}.pth'))
    # torch.save(D_A.state_dict(), os.path.join(save_path, f'model_discriminator_A_{epoch}.pth'))
    # torch.save(D_B.state_dict(), os.path.join(save_path, f'model_discriminator_B_{epoch}.pth'))

def plot_losses(G_losses, D_A_losses, D_B_losses, cycle_losses):
    """損失をプロットする関数"""
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_A_losses, label='Discriminator A Loss')
    plt.plot(D_B_losses, label='Discriminator B Loss')
    plt.plot(cycle_losses, label='Cycle Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Training Losses Over Time')
    plt.show()

def denormalize(tensor):
    """テンソルを非正規化して表示用に変換"""
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0, 1)
    return tensor

def save_image_grid(real_A, fake_B, recovered_A, real_B, fake_A, recovered_B, epoch, save_path):
    """変換結果を画像グリッドとして保存"""
    images = torch.cat([
        denormalize(real_A), denormalize(fake_B), denormalize(recovered_A),
        denormalize(real_B), denormalize(fake_A), denormalize(recovered_B)
    ], dim=0)

    torchvision.utils.save_image(
        images,
        os.path.join(save_path, f'results_epoch_{epoch}.png'),
        nrow=3,
        normalize=False
    )

def train(config, dataloader, device):
    # モデルの初期化
    G_AB = Generator().to(device)
    G_BA = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    # 損失関数の定義
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # オプティマイザの設定
    optimizer_G = optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=config.lr,
        betas=(config.b1, config.b2)
    )
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=config.lr, betas=(config.b1, config.b2))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=config.lr, betas=(config.b1, config.b2))

    # 損失の履歴を保存するリスト
    G_losses = []
    D_A_losses = []
    D_B_losses = []
    cycle_losses = []

    # 学習ループ
    for epoch in range(config.epoch, config.n_epochs):
        epoch_G_losses = []
        epoch_D_A_losses = []
        epoch_D_B_losses = []
        epoch_cycle_losses = []

        for i, batch in enumerate(dataloader):
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            # 偽の画像を生成
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)

            # Generatorの学習
            optimizer_G.zero_grad()

            # アイデンティティ損失
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) * config.lambda_identity

            # GAN損失
            loss_GAN_AB = criterion_GAN(D_B(fake_B), torch.ones_like(D_B(fake_B)))
            loss_GAN_BA = criterion_GAN(D_A(fake_A), torch.ones_like(D_A(fake_A)))
            loss_GAN = loss_GAN_AB + loss_GAN_BA

            # サイクル一貫性損失
            recovered_A = G_BA(fake_B)
            recovered_B = G_AB(fake_A)
            loss_cycle_A = criterion_cycle(recovered_A, real_A)
            loss_cycle_B = criterion_cycle(recovered_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) * config.lambda_cycle

            # 総損失
            loss_G = loss_GAN + loss_cycle + loss_identity
            loss_G.backward()
            optimizer_G.step()

            # Discriminator Aの学習
            optimizer_D_A.zero_grad()
            loss_real = criterion_GAN(D_A(real_A), torch.ones_like(D_A(real_A)))
            loss_fake = criterion_GAN(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A)))
            loss_D_A = (loss_real + loss_fake) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()

            # Discriminator Bの学習
            optimizer_D_B.zero_grad()
            loss_real = criterion_GAN(D_B(real_B), torch.ones_like(D_B(real_B)))
            loss_fake = criterion_GAN(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B)))
            loss_D_B = (loss_real + loss_fake) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()

            # 損失を記録
            epoch_G_losses.append(loss_G.item())
            epoch_D_A_losses.append(loss_D_A.item())
            epoch_D_B_losses.append(loss_D_B.item())
            epoch_cycle_losses.append(loss_cycle.item())

        # エポックごとの平均損失を記録
        G_losses.append(np.mean(epoch_G_losses))
        D_A_losses.append(np.mean(epoch_D_A_losses))
        D_B_losses.append(np.mean(epoch_D_B_losses))
        cycle_losses.append(np.mean(epoch_cycle_losses))

        # エポック終了時の進捗表示と画像保存
        print(f"[Epoch {epoch}/{config.n_epochs}] "
              f"[G loss: {G_losses[-1]:.4f}] "
              f"[D_A loss: {D_A_losses[-1]:.4f}] "
              f"[D_B loss: {D_B_losses[-1]:.4f}] "
              f"[Cycle loss: {cycle_losses[-1]:.4f}]")

        # モデルの保存
        save_models(G_AB, G_BA, D_A, D_B, epoch, config.model_save_path)

        # 最終バッチの変換結果を保存
        save_image_grid(
            real_A, fake_B, recovered_A,
            real_B, fake_A, recovered_B,
            epoch, config.model_save_path
        )

    # 訓練終了後に損失グラフを表示
    plot_losses(G_losses, D_A_losses, D_B_losses, cycle_losses)

    return G_losses, D_A_losses, D_B_losses, cycle_losses

# データ拡張の設定
transforms_ = transforms.Compose([
    transforms.Resize(int(Config().img_size * 1.12), transforms.InterpolationMode.BICUBIC),
    transforms.RandomCrop(Config().img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == "__main__":
    # GPUの確認
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # 設定の初期化
    config = Config()

    # データセットとデータローダーの初期化
    dataset = StatueHumanDataset(
        root_dir="/content/drive/MyDrive/チャレキャラ/new_dataset_241203",
        transform=transforms_
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )

    # 学習の実行
    print("Starting training...")
    G_losses, D_A_losses, D_B_losses, cycle_losses = train(config, dataloader, device)

    # 最終的な損失グラフの保存
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_A_losses, label='Discriminator A Loss')
    plt.plot(D_B_losses, label='Discriminator B Loss')
    plt.plot(cycle_losses, label='Cycle Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Final Training Losses')
    plt.savefig(os.path.join(config.model_save_path, 'final_losses.png'))
    plt.show()

    print("Training completed!")
    print(f"Models and results saved to: {config.model_save_path}")