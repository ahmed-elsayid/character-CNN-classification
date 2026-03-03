"""Model architecture."""

import torch.nn as nn


class chr_cnn(nn.Module):

  def __init__(self,config):

    super().__init__()

    self.text_len = config.text_len
    self.alpha_len = config.alpha_len
    self.n_class = config.num_classes
    self.use_relu= config.use_relu

    self.conv1 = nn.Sequential(
        nn.Conv1d(in_channels= self.alpha_len, out_channels= 256, kernel_size= 7),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size= 3)
    )

    self.conv2 = nn.Sequential(
        nn.Conv1d(in_channels= 256, out_channels= 256, kernel_size= 7),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size= 3)
    )

    self.conv3 = nn.Sequential(
        nn.Conv1d(in_channels= 256, out_channels= 256, kernel_size= 3),
        nn.ReLU()
    )

    self.conv4 = nn.Sequential(
        nn.Conv1d(in_channels= 256, out_channels= 256, kernel_size= 3),
        nn.ReLU()
    )

    self.conv5 = nn.Sequential(
        nn.Conv1d(in_channels= 256, out_channels= 256, kernel_size= 3),
        nn.ReLU()
    )

    self.conv6 = nn.Sequential(
        nn.Conv1d(in_channels= 256, out_channels= 256, kernel_size= 3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size= 3)
    )
    #  flatten layer

    self.flatten = nn.Flatten(start_dim= 1)

    # fully connected layers
    fc_input_dim = (((self.text_len - 96) // 27) * 256)

    self.fc1 = nn.Sequential(
            nn.Linear(in_features=fc_input_dim, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

    self.fc2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

    self.fc3  = nn.Sequential(
        nn.Linear(in_features= 1024, out_features= self.n_class),
    )


    self.init_weights()

  def init_weights(self):
    for layer in self.modules():

      if isinstance(layer, nn.Conv1d):
        nn.init.normal_(layer.weight, mean= 0, std= 0.05)

      if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, mean= 0, std= 0.05)


  def forward(self, X):

    X = self.conv1(X)
    X = self.conv2(X)
    X = self.conv3(X)
    X = self.conv4(X)
    X = self.conv5(X)
    X = self.conv6(X)

    X = self.flatten(X)

    X = self.fc1(X)
    X = self.fc2(X)
    y = self.fc3(X)

    return y


def create_model(config):
    """Create model from config."""
    return chr_cnn(config)
