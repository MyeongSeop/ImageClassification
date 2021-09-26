# Image Classification

* Semi-Supervised Learning for Image Classification
* Used pytorch & ResNet-50(pre-trained model from pytorch)
* Accuracy : 95.842%
* Kaggle leaderBoard : https://www.kaggle.com/c/dnn2021ssl/leaderboard


### Data

- Train : 5000 (Labeled), 35551(Unlabeled)
- Test : 10000
- Classes : 10
- Type : 32*32 image with RGB channels
- Transform : Resize, Normalize, RandomRation, Sharpness
- Examples

  ![data](https://user-images.githubusercontent.com/28529183/134809925-c00e4486-eef3-4d83-8980-4cd68f489369.JPG)


### Training (ResNet 50)

- ResNet 50

  ![ResNet50](https://user-images.githubusercontent.com/28529183/134810341-75f98d02-1910-4257-aec9-0026a856c613.JPG)

- Learning rate : 0.001
- Epoch : 20 / 8 (labeled data / unlabeled data -> pseudo-labeling)
- Batch size : 128
- Momentum : 0.9
- Optimizer : SGD
- Loss : Cross Entropy Loss
- Accuracy(Test) : 95.842%


### Pseudo Labeling

- Alpha Weight

  weight = ((step - 100) / (700 - 100)) * 3
  
  (step : initial value -> 100, increases 1 for every 50 batches)


### Other Models.....

- Personal CNN Model (model.py) : 87.333%
- DenseNet 161 : 90.4%
- EfficientNet b3 : 90.166%
