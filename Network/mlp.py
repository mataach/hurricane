import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(17, 50),
            nn.Sigmoid(),
            nn.Linear(50, 20),
            nn.Sigmoid(),
            nn.Linear(20, 5)
        )
        self.MLP.apply(self.init_weights)

    @staticmethod
    def init_weights(layer):
        """
        Initializes weights to uniform Xavier dist and biases to zero
        :param MLP layers
        :return None
        """
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward propagation for MLP model
        args:
            input: torch.Tensor - Float tensor with shape [batch_size,17]
        Returns
            logits:torch.Tensor - Float tensor with shape [batch_size,num_classes]
        """
        return self.MLP(input)


class Trainer:
    def __init__(self, epochs, model, criterion, optim):
        self.model = model
        self.epochs = epochs
        self.criterion = criterion
        self.optim = optim

    def train(self, train_loader):
        self.model.train()
        acc_stats, loss_stats = [], []
        for i in range(self.epochs):
            for data, target in train_loader:
                self.optim.zero_grad()
                logits = self.model.forward(input=data)
                loss = self.criterion(logits, target)
                loss.backward()
                self.optim.step()

                # Statistics
                accuracy = self.accuracy_computer(logits=logits, target=target)
                acc_stats.append(accuracy)
                loss_stats.append(loss.item())

            print('Epoch: ', i)
            print('Training acc: {0:0.4f}'.format(np.mean(np.array(acc_stats))))
            print('Training loss: {0:0.4f}'.format(np.mean(np.array(loss_stats))))
            print('---------------------------------')

    def test(self, test_loader, visualize=False):
        self.model.eval()
        acc_stats, loss_stats = [], []
        for data, target in test_loader:
            logits = self.model.forward(input=data)
            loss = self.criterion(logits, target)

            # Statistics
            accuracy = self.accuracy_computer(logits=logits, target=target)
            acc_stats.append(accuracy)
            loss_stats.append(loss.item())

        if visualize: return logits
        else:
            print('Testing acc: {0:0.4f}'.format(np.mean(np.array(acc_stats))))
            print('Testing loss: {0:0.4f}'.format(np.mean(np.array(loss_stats))))

    @staticmethod
    def accuracy_computer(logits: torch.Tensor, target: torch.LongTensor) -> float:
        """Computes the accuracy from softmax logits and target labels
        Args:
            logits: torch.Tensor [batch_size,num_classes]- Output of the forward method
            target: torch.LongTensor [batch_size,] - Target label class indices
        Returns:
            accuracy: float - Accuracy for the current batch of examples
        """
        return np.mean(np.argmax(logits.detach().numpy(), axis=1) == np.argmax(target.detach().numpy(), axis=1)).item()

    def visualize_confusion_matrix(self, y_test, test_loader):
        """
        Plot confusion matrix of true vs predicted labels
        :return: None
        """
        predictions = self.test(test_loader, visualize=True)
        cm = confusion_matrix(y_true=np.argmax(y_test, axis=1), y_pred=np.argmax(predictions.detach().numpy(), axis=1))
        # Visualize
        fig = plt.figure(figsize=(8, 8), dpi=150)
        ax = sns.heatmap(cm, vmax=None, square=True, annot=True, cmap='viridis')
        ax.set_title('Confusion Matrix', fontsize=16)
        ax.set_xlabel('Predicted label', fontsize=12)
        ax.set_ylabel('True label', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        plt.show()


