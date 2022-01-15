import cv2
import numpy as np
from tqdm import tqdm

# Модуль pickle реализует двоичные протоколы для сериализации и де-сериализации объектной структуры Python.
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchsummary

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt

from model import ClassificationModel
# Класс ClassificationModel используется для всех задач классификации текста, кроме классификации по нескольким меткам.

from dataset import CustomDataset
from utils import calcAccuracy

matplotlib.use('TkAgg')

if __name__ == "__main__":
	# Чтобы протестировать набор данных и наш загрузчик данных, в главной
	# функции нашего скрипта, мы создаем экземпляр созданного
	# CustomDataset и называем его dataset.
	dataset = CustomDataset()
	dataset_train, dataset_test = train_test_split(dataset, test_size=0.3) # 30% используем для тестирования

	data_loader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
	data_loader_test = DataLoader(dataset_test, batch_size=16, shuffle=True)

	# Вывод на экран изображение и label (метку)
	train_features, train_labels = next(iter(data_loader_train))
	train_features = dataset.getImgsTensors(train_features)

	print(f"Feature batch shape: {train_features.size()}")
	print(f"Labels batch shape: {train_labels.size()}")

	plt.title('Labels: ' + ', '.join([dataset.getName(i) for i in train_labels]))
	gridImgs = torchvision.utils.make_grid(train_features)
	# оставляем изображение в исходной цветовой гамме (три канала - цветное изображение)
	plt.imshow(cv2.cvtColor(gridImgs.permute(1, 2, 0).numpy() / 255, cv2.COLOR_BGR2RGB))
	# plt.show()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)

	model = ClassificationModel().to(device)
	print(model)
	torchsummary.summary(model, (3, 256, 256))

	# использовать классификационный Cross-Entropy loss и SGD с импульсом = 0.9.
	criterion = nn.CrossEntropyLoss() # делает для нас softmax
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # Реализует стохастический градиентный спуск
	# optimizer = optim.Adam(model.parameters(), lr=0.001)

	best_loss = np.inf
	best_accuracy = 0.0

	for epoch in range(30):
		model.train()
		epoch_loss = 0.0

		for i, data in enumerate(tqdm(data_loader_train, desc='[%d] Training batches' % (epoch + 1)), 0):
			# получаем входные данные; данные - это список [inputs, labels].
			inputs, labels = data
			labels = labels.to(device)
			inputs = dataset.getImgsTensors(inputs).to(device)

			# обнуляем параметр gradients:
			optimizer.zero_grad()

			# forward + backward + optimize:
			outputs = model(inputs)
			# exit()

			loss = criterion(outputs, torch.max(labels, 1)[0])
			loss.backward()

			optimizer.step()
			# вывести статистику
			epoch_loss += loss.item()

		print('[%d] Train loss: %.10f %.10f' % (epoch + 1, epoch_loss, epoch_loss / len(data_loader_train)))

		model.eval()
		epoch_loss = 0.0
		accuracy = 0.0

		with torch.no_grad():
			predictList = []
			targetList = []

			for i, data in enumerate(tqdm(data_loader_test, desc='[%d] Testing batches' % (epoch + 1)), 0):
				# получаем вводные данные:
				inputs, labels = data
				labels = labels.to(device)
				inputs = dataset.getImgsTensors(inputs).to(device)

				outputs = model(inputs)

				predictList += torch.max(outputs, 1)[1].tolist()
				targetList += labels.squeeze(1).tolist()

				accuracyDict = calcAccuracy(targetList, predictList)

				loss = criterion(outputs, torch.max(labels, 1)[0])

				epoch_loss += loss.item()

		print('[%d] Test loss: %.10f %.10f' % (epoch + 1, epoch_loss, epoch_loss / len(data_loader_test)))
		print('[%d] Test accuracy:' % (epoch + 1))

		print(*[f'\t{dataset.getName(i)}: {accuracyDict[i]:.2f}' for i in accuracyDict.keys()], sep='\n')

		accuracy = sum([accuracyDict[i] for i in accuracyDict.keys()]) / len(accuracyDict.keys())

		if best_accuracy < accuracy:
			print('New best model with average accuracy: ', accuracy)
			best_accuracy = accuracy
			best_model = pickle.loads(pickle.dumps(model))

		print(10 * '-')

	print('Тренировка завершена, наилучшая средняя точность: ', best_accuracy)

	print('Проверка наилучшей модели')

	# Вывод на экран изображение и label (метку).
	test_features, test_labels = next(iter(data_loader_test))
	test_labels = test_labels.to(device)
	test_features = dataset.getImgsTensors(test_features).to(device)

	outputs = best_model(test_features)
	outputs = torch.max(outputs, 1)[1]

	plt.title('Data labels: ' + ', '.join([dataset.getName(i) for i in test_labels]) + \
			  '\nPredicted labels: ' + ', '.join([dataset.getName(i) for i in outputs]))
	gridImgs = torchvision.utils.make_grid(test_features).cpu()
	plt.imshow(cv2.cvtColor(gridImgs.permute(1, 2, 0).numpy() / 255, cv2.COLOR_BGR2RGB))
	plt.show()
