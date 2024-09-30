import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import hydra
from omegaconf import DictConfig
import wandb
import requests
import os
import zipfile

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        return self.fc(x.view(-1, 28*28))

def download_data_from_s3(s3_url, local_data_path="data"):
    res = requests.get(s3_url)
    open('content.zip', 'wb').write(res.content)
    with zipfile.ZipFile('content.zip', 'r') as zip_ref:
        zip_ref.extractall(local_data_path)

def load_data(local_data_dir):
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root=os.path.join(local_data_dir, 'mnist_data'), 
        train=True, 
        download=False,  # Set to False since data is already downloaded
        transform=transform
    )
    data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    return data_loader

@hydra.main(version_base=None)
def train(cfg: DictConfig):
    wandb.login()
    wandb.init(project=cfg.wandb_project, config=dict(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    download_data_from_s3(cfg.s3.url, cfg.s3.local_data_path)
    data_loader = load_data(cfg.s3.local_data_path)
    model = SimpleModel().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    for epoch in range(cfg.training.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(data_loader)
        accuracy = 100 * correct / total
        wandb.log({"epoch": epoch, "loss": epoch_loss, "accuracy": accuracy})
        print(f"Epoch [{epoch+1}/{cfg.training.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    os.makedirs(os.path.join(cfg.s3.local_data_path, "checkpoints"), exist_ok=True)
    save_path = os.path.join(cfg.s3.local_data_path, "checkpoints", "final_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return save_path

if __name__ == "__main__":
    train()
