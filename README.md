# plant_disease_classifier
CNN built to predict the disease based on a picture

This is a small data science project

2021/01 : This model can predict :
- chlorosis
- powdery mildew
- downy mildew
- rust

## Model creation and fitting
The model was built using Google Images datasets. Thus the datasets will not be provided.

The code for building and fitting the model is available in **plant_disease_classifier_model.ipynb**

## Disease prediction
To predict a plant disease based on a picture, use **predict_plant_disease.py**

The command :
```
python predict_plant_disease.py -i "path_to_image.jpg"
```

You can use .jpeg or .jpg format

The output is a table that has the probability of the disease as rows
