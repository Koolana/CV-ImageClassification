import glob # Модуль glob находит все имена путей, соответствующие заданному шаблону
import torch
import cv2

from torch.utils.data import Dataset

class CustomDataset(Dataset):
	def __init__(self):
		self.imgs_path = 'dataset' # корневая папка, где находится датасет
		file_list = glob.glob(self.imgs_path + '/*')
		# print(file_list)

		self.data = []
		for class_path in file_list:
			class_name = class_path.split('/')[-1]
			for img_path in glob.glob(class_path + '/' +'*.jpg'):
				self.data.append([img_path, class_name])

		# print(*self.data, sep='\n')
		# записываем существующие классы
		self.class_map = {'Bread' : 0, 'Dessert' : 1, 'Meat' : 2, 'Soup' : 3}
		self.img_dim = (256, 256) # (32, 32)

	# функция, которая возвращает длину набора данных
	def __len__(self):
		return len(self.data)

	# функция, которая возвращает один обучающий пример
	def __getitem__(self, idx):
		img_path, class_name = self.data[idx]

		# сопоставление имени с числом
		class_id = self.class_map[class_name]

	    # преобразуем целочисленное значение class_id в тензор 
	    # также увеличиваем его размерность, ссылаясь на него как [class_id]. 
	    # Это необходимо для того, чтобы обеспечить возможность пакетной
	    # обработки данных в тех размерах, которые требуются torch.
		class_id = torch.tensor([class_id])

		return img_path, class_id

	def getImgsTensors(self, imgs_path):
		output_tensor = torch.tensor([], dtype=torch.float)

		for img_path in imgs_path:
			img = cv2.imread(img_path)
			# print(img.shape)
			img = cv2.resize(img, self.img_dim)

		    # преобразуем переменные в тензоры (torch.from_numpy позволяет
		    # преобразовать массив numpy в тензор)
			img_tensor = torch.from_numpy(img)
			# замены осей ((Каналы, Ширина, Высота))
			img_tensor = img_tensor.permute(2, 0, 1).float()

			img_tensor = img_tensor.unsqueeze(0)

			output_tensor = torch.cat((output_tensor, img_tensor))

		return output_tensor

	def getName(self, value):
	    for k, v in self.class_map.items():
	        if v == value:
	            return k
