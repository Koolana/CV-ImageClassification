# CV-ImageClassification

## Code requirements
numpy: pip3 install numpy

opencv: pip3 install opencv-python

torch: pip3 install torch

glob: pip3 install glob

## File "main.py"
Создание пользовательского набора данных и загрузчика данных в Pytorch подробно описано в [the documentation](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

Мы определяем функцию init для инициализации наших переменных. Переменная self.imgs_path содержит базовый путь к нашей папке "dataset".

Мы создали список со всеми нашими данными, мы начинаем кодировать функцию для __len__(), которая является обязательным для объекта Torch Dataset.

Размер нашего набора данных - это просто количество отдельных изображений, которые у нас есть, которое можно получить через длину списка self.data. (Torch внутренне использует эту функцию для понимания размера набора данных в своем dataloader, чтобы вызвать функцию __getitem__() с индексом в пределах этого размера набора данных).

## Dataset
Структура папок выглядит следующим образом. У нас есть папка Project, которая содержит код main.py и код model.py, и папка под названием "dataset". Эта папка под названием "dataset" - это папка dataset, которая содержит 4 вложенные папки под названиями Soup, Dessert, Meat и Bread.


[Ссылка на набор данных здесь](https://drive.google.com/drive/folders/1fkSZmSQo_W6Jz3Jb5R0bWwQKKH1Pn2x0?usp=sharing)
