This README file is much more like the report.

#task.py
  
I create the Model class when it is called, it reads 3 csv files: chefmozparking.csv, geoplaces2.csv, rating_final.csv.
After merging the corresponding datasets, I select "price", "parking_lot", "smoking_area", "other_services", "dress_code", "accessibility" features and "rating" as a target label as mentioned in the task.
I check the feature values but there are no inconsistencies which do not make sense. I just change some feature values to use as a column name according to the variable name standards. 
In the next stage, I convert the categorical values into dummy/indicator values, get the train and test splits. I just select 2 vectors as a test dataset. I build, fit and serialize the model to reconstruct later. 
In the get_prediction method, I prepare the vector as a json, request to the api and get the response as a json.

#main.py

I deserialize the model, get the json, predict the label and send back as a json.

#pipeline.py

I build the pipeline which writes the predicted value to the prediction.txt file.

#predictions.txt

contains the predicted labels.

How to run main.py: python main.py

How to run pipeline.py:  python pipeline.py Prediction  --local-scheduler

I have also upload the mindtitan_task.ipynb which contains all stages(data transformation, model training and gettin the prediction)
