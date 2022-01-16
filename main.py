import cv2
import numpy as np
import tqdm as tq

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
from trainer import Trainer

matplotlib.use('TkAgg')

if __name__ == "__main__":
	# Чтобы протестировать набор данных и наш загрузчик данных, в главной
	# функции нашего скрипта, мы создаем экземпляр созданного
	# CustomDataset и называем его dataset.

	# Преднамеренно разбиваем выборку на train, test, validation.
	# Это нужно для того, чтобы проводить тестирование на том, что вообще не связано с обучением модели.
	dataset = CustomDataset()
	dataset_train, dataset_test = train_test_split(dataset, test_size=0.2)
	dataset_train, dataset_val = train_test_split(dataset_train, test_size=0.2)

	data_loader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
	data_loader_test = DataLoader(dataset_test, batch_size=16, shuffle=True)
	data_loader_val = DataLoader(dataset_val, batch_size=16, shuffle=True)

	# Вывод на экран изображение и label (метку)
	train_features, train_labels = next(iter(data_loader_train))
	train_features = dataset.getImgsTensors(train_features)

	plt.title('Labels: ' + ', '.join([dataset.getName(i) for i in train_labels]))
	gridImgs = torchvision.utils.make_grid(train_features)
	# оставляем изображение в исходной цветовой гамме (три канала - цветное изображение)
	plt.imshow(cv2.cvtColor(gridImgs.permute(1, 2, 0).numpy() / 255, cv2.COLOR_BGR2RGB))
	plt.show()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Used device:", device)

	model = ClassificationModel().to(device)
	# сжимаем изображение до размера 128 на 128. Так и нейронка меньше выйдет, и обучение пройдет гораздо быстрее.
	torchsummary.summary(model, (3, 128, 128))

	trainer = Trainer(model, data_loader_train, data_loader_test, data_loader_val,
					  device, getTensorsFunc=dataset.getImgsTensors, tqdm=tq.tqdm)

	best_loss = np.inf
	best_accuracy = 0.0

	for epoch in range(15):
		epoch_loss, _ = trainer.train(epoch + 1)

		print('[%d] Train loss: %.10f %.10f' % (epoch + 1, epoch_loss, epoch_loss / len(data_loader_train)))

		####################################################

		epoch_loss, accuracyDict = trainer.test(epoch + 1)

		print('[%d] Test loss: %.10f %.10f' % (epoch + 1, epoch_loss, epoch_loss / len(data_loader_test)))

		accuracy = sum([accuracyDict[i] for i in accuracyDict.keys()]) / len(accuracyDict.keys())
		print('[%d] Test average accuracy: %.2f' % (epoch + 1, accuracy))

		print('[%d] Test accuracy in classes:' % (epoch + 1))
		print(*[f'\t{dataset.getName(i)}: {accuracyDict[i]:.2f}' for i in accuracyDict.keys()], sep='\n')

		####################################################

		if best_accuracy < accuracy:
			print('New best model with average accuracy: ', accuracy)
			best_accuracy = accuracy
			best_model = pickle.loads(pickle.dumps(model))

		print(40 * '-')

	print('Тренировка завершена, наилучшая средняя точность: ', best_accuracy)

	####################################################

	print('\nПроверка наилучшей модели')

	accuracyDict, confusionMatrix = trainer.validation(best_model)

	test_accuracy = sum([accuracyDict[i] for i in accuracyDict.keys()]) / len(accuracyDict.keys())
	print('Validation average accuracy: %.2f' % (test_accuracy))
	print('Accuracy in classes:')
	print(*[f'\t{dataset.getName(i)}: {accuracyDict[i]:.2f}' for i in accuracyDict.keys()], sep='\n')

	print('Confusion matrix:')
	print(confusionMatrix)

	####################################################

	# Вывод на экран изображение и label (метку).
	test_features, test_labels = next(iter(data_loader_val))
	test_labels = test_labels.to(device)
	test_features = dataset.getImgsTensors(test_features).to(device)

	outputs = best_model(test_features)
	outputs = torch.max(outputs, 1)[1]

	plt.title('Data labels: ' + ', '.join([dataset.getName(i) for i in test_labels]) + \
			  '\nPredicted labels: ' + ', '.join([dataset.getName(i) for i in outputs]))
	gridImgs = torchvision.utils.make_grid(test_features).cpu()
	plt.imshow(cv2.cvtColor(gridImgs.permute(1, 2, 0).numpy() / 255, cv2.COLOR_BGR2RGB))
	plt.show()
