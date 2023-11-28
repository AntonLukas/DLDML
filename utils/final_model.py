import torch.nn as nn


class Binnary_CNN(nn.Module):
    def __init__(self, dropout_rate):
        super(Binnary_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=(3, 3), padding=(0, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout_rate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=(3, 3), padding=(0, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout_rate)
        )

        self.out = nn.Linear(96, 1)
        self.sigm = nn.Sigmoid()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        # Prefer Kaiming He weights initialization over Xavier Glorot for ReLU activation.
        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
        # Xavier Glorot used for weight initialization over Kaiming He for Sigmoid activation.
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, nn_input):
        x = self.conv1(nn_input)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)

        x = self.out(x)
        output = self.sigm(x)

        return output
