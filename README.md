# Классификация цветных изображений четырех видов еды с использованием сверточной нейронной сети (CNN)

## Начало работы

### Зависимости
Данный проект написан с помощью **Python 3** и был протестирован на совместимость со следующими библиотеками:

```
numpy==1.21.2
torch==1.10.0
torchvision==0.11.2
torchsummary==1.5.1
opencv-contrib-python==4.5.3.56
tqdm==4.62.3
scikit-learn==0.24.2
matplotlib==3.4.2
```

- __glob__: позволяет нам легко находить пути к данным внутри вложенных папок;

- __cv2__: используется в качестве библиотеки обработки изображений для чтения и предварительной обработки изображений;

- __numpy__: используется для матричных операций;

- __torch__: используется для создания классов Dataset и Dataloader, а также для преобразования данных в тензоры.

### Установка

```
git clone https://github.com/Koolana/CV-ImageClassification.git
```

## Запуск

Для запуска обучения и проверки модели на датасете необходимо сначала скопировать датасет в папку **datasets**, так, чтобы каждая вложенная папка соответствовала одному определенному классу, а затем запустить следующую команду из корня данного проекта:

```
python3 main.py
```

Подробное описание работы с примерами и метрики точности представлены в [блокноте Jupyter](demo.ipynb)

## Датасет

Для проверки работоспособности модели и оценки точности был использован [датасет изображений еды](https://drive.google.com/drive/folders/1fkSZmSQo_W6Jz3Jb5R0bWwQKKH1Pn2x0?usp=sharing), в котором присутствует 4 класса: Soup, Dessert, Meat и Bread. Данный датасет помещается в папку **datasets**, где каждая подпапка соответсвует определенному классу. Ниже представлен пример данных из этого датасета:

<img width="581" alt="before" src="https://user-images.githubusercontent.com/37844731/149667285-365c73c4-c412-4edf-822c-be855401c249.png">

В нашей работе мы использовали разделение на следующие выборки: **тренировочная, тестовая и валидационная**.

## Описание модели
Обучение модели глубокого обучения требует от нас преобразования данных в формат, который может быть обработан моделью. 

Нейронные сети могут быть построены с использованием пакета torch.nn.
torch.nn зависит от *autograd* в определении моделей и их дифференцировании. *nn.Module* содержит слои и метод *forward(input)*, который возвращает output.
Типичная процедура обучения для нейронной сети следующая:

1. Определение нейронной сети, которая имеет некоторые изучаемые параметры (или веса);
2. Итерация по набору входных данных;
3. Обработка ввода через сеть;
4. Рассчет потери (насколько далеки результаты от правильных);
5. Распространение градиентов обратно на параметры сети;
6. Обновление весов сети, как правило, используя простое правило обновления:
weight = weight - learning_rate * gradient.

Использована нейронная сеть для приема 3-канального изображения.

Архитектура модели представлена на изображении ниже:

<img width="581" alt="summary" src="https://user-images.githubusercontent.com/90565598/149660120-f15326a5-106a-45fb-a0e1-fa2f77e82acb.png">

## Обучение модели

Данная модель была обучена на 20 эпохах с использованием функции потерь **Cross entropy** и оптимизатора **SGD**, все изображения в датасете сжаты до размера 128х128.

График зависимости значения функции потерь от номера эпохи для тренировочной и тестовой выборок:

<img width="581" alt="loss" src="https://user-images.githubusercontent.com/90565598/149660161-7e644143-75c1-4a78-9c36-e4109a2fb0b2.png">

График зависимости средней точности классификации от номера эпохи для тренировочной и тестовой выборок:

<img width="581" alt="accuracy" src="https://user-images.githubusercontent.com/90565598/149660013-11d7b5f6-7f73-4411-9146-a373afc67e8e.png">

Итоговая обученная модель **trainedModel.pt** находится в папке **models**.

## Результаты

Для нашей модели получены следующие результаты точности на валидационной выборке:

```
Validation average accuracy: 0.79
Accuracy in classes:
	Bread: 0.86
	Dessert: 0.54
	Meat: 0.87
	Soup: 0.90
```

Так же получена следующая матрица ошибок для валидационной выборки:

<img width="581" alt="accuracy" src="https://user-images.githubusercontent.com/37844731/149667207-5a0469a0-138b-4f42-bc14-1ef9aa94dec0.png">

Примеры неправильной классификации изображений:

<img width="581" alt="predImgs" src="https://user-images.githubusercontent.com/37844731/149667152-a5b2152b-be8b-434c-adc0-b443689e88e1.png">

Результат на случайной валидационной выборке:

<img width="619" alt="after" src="https://user-images.githubusercontent.com/90565598/149660214-94d359f3-2bde-4a08-8f59-4b78a4846e4c.png">

Подробный отчет по данной работе представлен в файле [блокнота Jupyter](demo.ipynb).

Авторы работы: Андрейчик Николай и Шутова Ксения.
