# CV-ImageClassification

## Code requirements
numpy: pip3 install numpy

opencv: pip3 install opencv-python

torch: pip3 install torch

glob: pip3 install glob

## File "main.py"
Creating a custom dataset and data loader in Pytorch is described in detail in [the documentation](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

We define the init function to initialize our variables. The variable self.imgs_path contains the base path to our "dataset" folder.

We have created a list with all our data, we start coding the function for __len__(), which is mandatory for a Torch Dataset object.

The size of our dataset is just the number of individual images we have, which can be obtained through the length of the self.data list. (Torch internally uses this function to understand the size of the dataset in its dataloader, to call the __getitem__() function with an index within this dataset size).

## Dataset
The folder structure is as follows. We have the Project folder that contains the code main.py and code model.py, and a folder called "dataset". This folder called "dataset" is the dataset folder that contains 4 subfolders inside it called Soup, Dessert, Meat and Bread.

[Link to dataset here](https://drive.google.com/drive/folders/1fkSZmSQo_W6Jz3Jb5R0bWwQKKH1Pn2x0?usp=sharing) (Здесь представлена ссылка на датасет).
