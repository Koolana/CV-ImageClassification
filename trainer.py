import torch
import torch.nn as nn
import torch.optim as optim

from utils import calcAccuracy

class Trainer():
    def __init__(self, model, trainDataLoader, testDataLoader, valDataLoader, device, getTensorsFunc=None, tqdm=None):
        self.model = model

        # использовать классификационный Cross-Entropy loss и SGD с импульсом = 0.9.
        self.criterion = nn.CrossEntropyLoss() # делает для нас softmax
        self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # Реализует стохастический градиентный спуск

        self.trainDataLoader = trainDataLoader
        self.testDataLoader = testDataLoader
        self.valDataLoader = valDataLoader

        self.device = device

        self.getTensorsFunc = getTensorsFunc
        self.tqdm = tqdm

    def train(self, epoch):
        self.model.train()
        train_loss = 0.0
        predictList = []
        targetList = []

        for i, data in enumerate(self.tqdm(self.trainDataLoader, desc='[%d] Training batches' % (epoch)) if self.tqdm is not None else self.trainDataLoader, 0):
            # получаем вводные данные
            inputs, labels = data
            labels = labels.to(self.device)
            inputs = self.getTensorsFunc(inputs).to(self.device)

            # обнуляем параметр gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(inputs)

            loss = self.criterion(outputs, torch.max(labels, 1)[0])
            loss.backward()

            self.optimizer.step()

            train_loss += loss.item()

            predictList += torch.max(outputs, 1)[1].tolist()
            targetList += labels.squeeze(1).tolist()

        accuracyDict = calcAccuracy(targetList, predictList)

        return train_loss, accuracyDict

    def test(self, epoch):
        self.model.eval()
        test_loss = 0.0
        test_accuracy = 0.0

        with torch.no_grad():
            predictList = []
            targetList = []

            for i, data in enumerate(self.tqdm(self.testDataLoader, desc='[%d] Testing batches' % (epoch)) if self.tqdm is not None else self.testDataLoader, 0):
                # получаем вводные данные
                inputs, labels = data
                labels = labels.to(self.device)
                inputs = self.getTensorsFunc(inputs).to(self.device)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, torch.max(labels, 1)[0])

                test_loss += loss.item()

                predictList += torch.max(outputs, 1)[1].tolist()
                targetList += labels.squeeze(1).tolist()

        accuracyDict = calcAccuracy(targetList, predictList)

        return test_loss, accuracyDict

    def validation(self, model):
        model.eval()

        with torch.no_grad():
            predictList = []
            targetList = []

            for i, data in enumerate(self.tqdm(self.valDataLoader, desc='Batches') if self.tqdm is not None else self.valDataLoader, 0):
                inputs, labels = data
                labels = labels.to(self.device)
                inputs = self.getTensorsFunc(inputs).to(self.device)

                outputs = model(inputs)

                predictList += torch.max(outputs, 1)[1].tolist()
                targetList += labels.squeeze(1).tolist()

        accuracyDict = calcAccuracy(targetList, predictList)

        return accuracyDict
