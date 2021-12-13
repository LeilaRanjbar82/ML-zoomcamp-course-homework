# PROJECT DESCRIPTION
This project performed as part of the _ML Zoomcamp Course_, Capstone Project. This course is conducted by [Alexey Grigorev](https://bit.ly/3BxeAoB)

# Yoga Posture Image Classification By Convolutional Neural Network(CNN)

## 1. Task Description
The yoga posture images are provided in 6 classes. I chose this dataset to apply CNN, check different models and actiovation functions and tuning the model parameters and augmentation. In my experience it is a good dataset for learning and testing CNN. 

## 2. Data

### 2.1. Data Refrence
This Model was built using [kaggle Dataset](https://www.kaggle.com/suradechk/yoga-posture-cleaned).

Yoga posture dataset obtain from three following sites and applied basic data cleaning manually:
* Open source dataset from [kaggle Dataset](https://www.kaggle.com/general/192938)
* 3D synthetic dataset from [aurencemoroney](https://laurencemoroney.com/2021/08/23/yogapose-dataset.html)
* Yoga-82 dataset from [google](https://sites.google.com/view/yoga-82/home)

(P.S. Kaggle dataset is a clean dataset)

### 2.2. Get the Data
(You can find codes in DataPreparation.ipynb)

Before starting, you need to have the opendatasets library installed in your system. If it’s not present in your system, use Python’s package manager pip and run:

```
    !pip install opendatasets
```

in a Jupyter Notebook cell. Python’s opendatasets library is used for downloading open datasets from platforms such as Kaggle.

The process to Download is as follows:

1. Import the opendatasets library

```
    import opendatasets as od
```

2. Now use the download function of the opendatasets library, which as the name suggests, is used to download the dataset. It takes the link to the dataset as an argument.

```
    od.download("https://www.kaggle.com/suradechk/yoga-posture-cleaned")
```

3. On executing the above line, it will prompt for Kaggle username. Kaggle username can be fetched from the **Account** tab of the **My Profile** section.

4. On entering the username, it will prompt for Kaggle Key. Again, go to the **Account** tab of the **My Profile** section and click on **Create New API Token**. This will download a _kaggle.json_ file.

5. On opening this file, you will find the _username_ and _key_ in it. Copy the key and paste it into the prompted Jupyter Notebook cell. The content of the downloaded file would look like this:

    `{"username":<KAGGLE USERNAME>,"key":"<KAGGLE KEY>"}`

6. A progress bar will show if the dataset is downloaded completely or not.

7. After successful completion of the download, a folder will be created in the current working directory of your Jupyter Notebook. This folder contains our dataset.

`[REF: https://www.analyticsvidhya.com/blog/2021/04/how-to-download-kaggle-datasets-using-jupyter-notebook/]`

**P.S.** I have been uploaded the dataset in [Capstone Project](https://github.com/LeilaRanjbar82/ML-zoomcamp-course-homework/tree/main/CapstoneProject) (complete and splitted in the 3 folders, train, valisation, and test). You don't need to follow the structure. I put it here just for clarifying the process.

### 2.3. Data Detail and Exploratory Data Analysis
(You can find codes in notebook.ipynb)
Yoga Posture Dataset consist of 2036 pictures in 6 different folder:

|**ID** |**Pose** | **#Images** |
|---|---|---|
|1|Chair|206|
|2|Cobra|643|
|3|Downdog|430|
|4|Goddes|209|
|5|Tree|362|
|6|Warrior|183|

![image](https://user-images.githubusercontent.com/58926709/145874264-0540ceaa-d29a-47a3-824c-e936aaca8212.png)

The image size of each class are plotted: 

![image](https://user-images.githubusercontent.com/58926709/145874556-cc383984-06db-46fb-94c5-175be23b9ac0.png), ![image](https://user-images.githubusercontent.com/58926709/145874582-2a7072e4-0015-4460-b6fe-badd7859f7f2.png)

![image](https://user-images.githubusercontent.com/58926709/145874645-c47153ea-4325-4bc8-9d74-1c38951afd59.png), ![image](https://user-images.githubusercontent.com/58926709/145874674-4569988c-40a0-4e24-b313-3c4d619b2565.png)

![image](https://user-images.githubusercontent.com/58926709/145874705-ae8e0912-9cfa-4318-9ce1-91d9ef10dfb0.png), ![image](https://user-images.githubusercontent.com/58926709/145874750-d9c1dac9-2830-4f40-bd8d-74af40ea9c3c.png)

Most of the image sizes are less than (1000 * 1000) pixels, just for _Cobra_ and _Downdog_ since thay have more images the sizes reached to (2000 * 2000) pixels.
There are some outlayer in each set, I didn't clean them.

In notebook I display some pictures to show the diversity of the images. They 8 first picture of each class. Following are few of them, one from each class.

![image](https://user-images.githubusercontent.com/58926709/145875392-4595ad42-d0dc-4915-b4a1-d8dfe453570c.png)
![image](https://user-images.githubusercontent.com/58926709/145875650-1f516e87-c0fe-42b5-81fb-3d74a8990322.png)
![image](https://user-images.githubusercontent.com/58926709/145875680-1a7fdd9a-e21a-4443-b477-4db182929e4c.png)

![image](https://user-images.githubusercontent.com/58926709/145875730-2b9a6ac4-07b2-4b2c-bf2c-e7164bf18e42.png)
![image](https://user-images.githubusercontent.com/58926709/145875778-92d86bfc-4d37-40df-84e8-3d24bb9f80fc.png)
![image](https://user-images.githubusercontent.com/58926709/145875814-7af865d2-9fe5-4f97-a3aa-54acb719711d.png)



### 2.4. Split data in Train, Test, Validation
(You can find codes in DataPreparation.ipynb)

For the part, I used two packages `os` and `shutil`.

`os.mkdir` is used to create the destination directories:

**--train**
* chair
* cobra
* downdog
* goddess
* tree
* warrior
    
**--validation**
* chair
* cobra
* downdog
* goddess
* tree
* warrior
    
**--test**
* chair
* cobra
* downdog
* goddess
* tree
* warrior

First I rename the files due to their classe by `os.rename`.

Then, `shutil.copy` is used to copy the file from source to destination folders as follow:

from each pose folder (chair, cobra, ...):
* The first 60% of images were copied to _train/pose_ folder
* The next 20% of images were copied to _validation/pose_ folder
* The rest were copied to _test/pose_ folder

## 3. Create Model
(You can find codes for 3.1 to 3.5 in notebook.ipynb)
**TensorFlow** is a library for ML and AI, and **Keras** from tensorfolw provides a Python interface for TensorFlow. In **keras.layers** you can find different layers to creat your model. more info in [keras layers](https://keras.io/api/layers/)
To classify the yoga images I used 7 model by combining layers and changing activation functions and optimizers.
### 3.1. Generate Dataset
I used **Xception** package from **keras.application** to preprocess the data. The Specifications are:
* Image size = (150,150)
* Batch size = 20
* Without Shuffilng
### 3.2. Training Different Model
1. The first model was a simple model. the layers descriptions are as follows:
* The base model is **Xception** with **imagenet** wight. Since in Keras, the top of a CNN is the dense layers, and the bottom is the convolutional part. I set the `include_top` to `False` to replace the Xception top.
* To avoid retraining the convolutional part of the network we have to set the `base_model.trainable` to `False`
* `keras.Input()` is an object that defines the shape of the input of our final model.
* A pooling layer was created by `keras.layers.GlobalAveragePooling2D()` to get proper vector representations of the input.
* The output layer is the Dense layer with output equals to number of classes, `keras.layers.Dense(6)`
* By `keras.model(inputs, outputs)` the model is defined.
* Dense layer is not trained. When initialized, its weights are set to random values. So, the optimizer, `keras.optimizers.Adam` is used to train the values of the weights by changing them on each training iteration in a way to creat output of the network make sense.
* `keras.losses.CategoricalCrossentropy` is used to prepare loss function
* We tied our model to optimizer, loss function and the metrics, interested in tracking, with `model.compile()`.
### 3.3. Tuning Parameter
#### 3.3.1. Learning Rate
#### 3.3.2. Inner size
#### 3.3.3. Drop Rate
### 3.4. Augmentation
### 3.5. Choosing the best model and Checkpointing
### 3.6. Test the model
### 3.7. Preparing Script
#### 3.7.1. Train
#### 3.7.2. Predict
#### 3.7.3. Predict_test and test the model by gunicorn
#### 3.7.4. Create pipfile and pipfile.lock
#### 3.7.5. Change format from .h5 to .tflite
#### 3.7.6. Create Lambda Function
#### 3.7.7. Create Docker file
### 3.8. Deployemeny
#### 3.8.1. Deploy and test the model locally
#### 3.8.2. Deploy to cloud









 
