import pytorch_lightning as pl
import mlflow.pytorch
from pytorch_lightning.loggers import MLFlowLogger
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define a simple feedforward neural network
class Net(pl.LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.functional.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

# Define a PyTorch Lightning dataloader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize MLflow logger
mlflow_logger = MLFlowLogger(experiment_name="mnist_experiment")

# Create a PyTorch Lightning trainer with MLflow logger
trainer = pl.Trainer(max_epochs=5, logger=mlflow_logger)

# Create the LightningModule and start training
model = Net()
trainer.fit(model, train_loader)

# Log the PyTorch model and hyperparameters to MLflow
with mlflow.start_run() as run:
    mlflow.log_params(model.hparams)
    mlflow.pytorch.log_model(model, "models")
