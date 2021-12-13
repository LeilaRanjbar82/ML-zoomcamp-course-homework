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
(You can find codes for 3.1 to 3.6 in notebook.ipynb)
**TensorFlow** is a library for ML and AI, and **Keras** from tensorfolw provides a Python interface for TensorFlow. In **keras.layers** you can find different layers to creat your model. more info in [keras layers](https://keras.io/api/layers/)
To classify the yoga images I used 7 model by combining layers and changing activation functions and optimizers.

### 3.1. Generate Dataset
I used **Xception** package from **keras.application** to preprocess the data. The Specifications are:
* Image size = (150,150)
* Batch size = 20
* Without Shuffilng

### 3.2. Training Different Model
**3.2.1.** The first model was a simple model. The layers descriptions are as follows:
* The base model is **Xception** with **imagenet** wight. Since in Keras, the top of a CNN is the dense layers, and the bottom is the convolutional part. I set the `include_top` to `False` to replace the Xception top.
* To avoid retraining the convolutional part of the network we have to set the `base_model.trainable` to `False`
* `keras.Input()` is an object that defines the shape of the input of our final model.
* A pooling layer was created by `keras.layers.GlobalAveragePooling2D()` to get proper vector representations of the input, reduce the size of the feature map.
* The output layer is the Dense layer with output equals to number of classes, `keras.layers.Dense(6)`.
* By `keras.model(inputs, outputs)` the model is defined.
* Dense layer is not trained. When initialized, its weights are set to random values. So, the optimizer, `keras.optimizers.Adam` is used to train the values of the weights by changing them on each training iteration in a way to creat output of the network make sense.
* `keras.losses.CategoricalCrossentropy` is used to prepare loss function.
* We tied our model to optimizer, loss function and the metrics, interested in tracking, with `model.compile()`.
**3.2.2.** The inner layer with output size = 100 and _ReLU_ activation function is added to  previous model, `keras.layers.Dense` 
**3.2.3.** `keras.layers.Flatten` is applied after vectorization.
**3.2.4.** Change the optimizer from _ADAM_ to _SGD_.
**3.2.5.** Use _ADAM_ optimizer. The output Dense layer activation function was set to _softmax_, so the loss function `from_logits` was changed to `False`.
**3.2.6.** _MaxPooling2D_ is used instead of _GlobalAveragePooling2D_ in pooling layer. The result was't good at all.
**3.2.7.** Using _softmax_ activation fuction for the output layer and _SGD_ for Optimization. _ADAM_ optimizer performed better.
**3.2.8.** Add second Inner Layer with output size = 500 and _SeLU_ activation function.

How ever in deep learning two runs never follow each other, the best result dur to these model was for **3.2.5.**. The next step is tuning parameter.

### 3.3. Tuning Parameter
Three parameter were choosed to tune, learning rate of optimizer, inner size of the inner dense layer, and drop rate for droping layer which is added before output layer.

#### 3.3.1. Learning Rate
Between four values of `0.0001, 0.001, 0.01, 0.1` the best performance was for 0.001.

![image](https://user-images.githubusercontent.com/58926709/145892853-8ee15b04-722a-4c9e-8183-ffdd1d8d25ba.png)

#### 3.3.2. Inner size
The best inner size among `50, 100, 200, 500`, shows the competance between `100` and `500`. 

![image](https://user-images.githubusercontent.com/58926709/145893386-2e6c859f-fc1e-41de-a4e6-00de92e11ae0.png)

So retraining the model with more epochs = 15, showed that the `size = 500` is better.

![image](https://user-images.githubusercontent.com/58926709/145893422-9e1fba90-2bce-450d-b070-5318d6a1c92c.png)

#### 3.3.3. Drop Rate
Due to the dataset, we may experience the overfitting. Using dropping layer to avoid it. On each epoch, different nodes were freezed, which means that their values are not updated.
The best drop rate among `0.0, 0.2, 0.5, 0.8` values was 0.2.

![image](https://user-images.githubusercontent.com/58926709/145894111-ed065492-e3e2-41f4-b920-c460cc718023.png)

### 3.4. Augmentation
Augmentation is used to prepare poor images, have better performance. Images were changed in size, direction and also rotation.
The result of Augmentation was not satisfying. However, changing the parameters and also parameters values may result in better performance. 
Note that the augmentation is only applied to train set.

### 3.5. Choosing the best model and Checkpointing
As mentined before the best result refered to a model with one inner dense layer, with _ADAM_ optimizer and _softmax_ output activation function. Using checkpointing to select the best model and save it due to validation accuracy. The model file is `yoga_08_0.862.h5`.

### 3.6. Test the model
The tuned model is test for a downdog image from test folder. The result:
```
{'chair': 7.7301534e-05,
 'cobra': 7.975984e-05,
 'downdog': 0.9998079,
 'goddess': 1.6248295e-07,
 'tree': 1.1697032e-06,
 'warrior': 3.3702236e-05}
```
The downdog value is the highest.

### 3.7. Preparing Script
To use the model I prepared different script. Before using script you have to download the dataset from data folder, save it in a capstone project folder in your computer and unzip it.
#### 3.7.1. Train
Train the model and save the models using checkpoint. Now, we have to choose the best model manually.
#### 3.7.2. Predict
Predict the output of the model. It is saves using flask.
In terminal you can use this command to run it.
```
gunicorn --bind 0.0.0.0:8080 predict:app
```
#### 3.7.3. Predict_test and test the model by gunicorn
The test file is the same as notebook test. After runnig predict, you can use following command in new terminal:
```
python predict_test.py
```
The result will be the same as 3.6.

#### 3.7.4. Create pipfile and pipfile.lock
Create pipfiles to have all required file for create model, because for predicting we remove tensorflow dependencies of the model. Tensorflsion ow package is a large package. We use the lighter version called _tflite_. But I prepared pipfile and pipfile.lock with numpy, tensorflow, flask and gunicorn packages.
To create pipfiles, first we need to install `pipenv`. The command is:
```
pip install pipenv
```
then we install packages as follows:
```
pipenv install numpy tensorflow flask gunicorn
```

#### 3.7.5. Change format from .h5 to .tflite
The script `tflitemodel.py` is used to convert the model. However I put both `.h5` and `.tflite` model in the directory in github.

#### 3.7.6. Create Lambda Function
Lambda function perform like predict file but it is used for serverless AWS deployement, however we can test it locally.
The process is:
```
ipython

[1]: import lambda_function
[2]: event = {'url':'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSIPLzGVtV2MDVS8PSUsr7WpM-L91TEVxjp1Q&usqp=CAU'}
[3]: lambda_function.lambda_handler(event, None)
```

#### 3.7.7. Create Docker file
For creating Docker file the requirements are:
```
using python from public.ecr.aws/lambda/python:3.8
installing keras-image-helper
installing tflite_runtime from https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl(a special version accepted by aws)
model: yoga_08_0.862.tflite .
lambda_function.py .

command: lambda_function.lambda_handler
```
#### 3.7.8. Containerization
follow the commands:
```
docker build -t yoga-model .
```
### 3.8. Deployemeny
#### 3.8.1. Deploy and test the model locally
Run the docker image
```
docker run -it --rm -p 8080:8080 yoga-model:latest
```
In another terminal follow the next command
```
python test.py
```
The result will be
```
{'chair': 0.012845357,
 'cobra': 0.017517198,
 'downdog': 0.9258903,
 'goddess': 0.008606592,
 'tree': 0.008768323,
 'warrior': 0.026372135}
 ```
 
 you can also find the test result in test.ipynb.

#### 3.8.2. Deploy to cloud









 
