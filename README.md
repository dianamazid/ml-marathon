Using both Jupyter notebook and python scripts for this project. While both Jupyter notebooks and Python scripts execute Python code, they serve different purposes and are used in different contexts. Notebooks are interactive, great for exploration and documentation, while scripts are sequential and suitable for automation and integration into larger workflows. 

Designing a Marathon Training Program: Using data from past marathon runners, I built a model to design personalized training programs. Features include the runner's age, gender, previous running experience, training logs, heart rate, and other physiological data.

These are the following steps I used for creating this project: 

Step 1: Data Collection

I used the following dataset for my project: Marathon time Predictions 

Open marathon_time_predictions.csv

marathon_time_predictions.csv

Step 2: Data Preprocessing

I loaded the dataset into a pandas DataFrame and cleaned the data to handle missing values, as well as normalize/scale features and encode categorical variables

Here is an example of my file: 

Open data_preprocessing.py

Open 01_data_preprocessing.ipynb

After running these scripts, a new data set was generated, which is as follows: 

Open preprocessed_marathon_data.csv

Step 3: Exploratory Data Analysis (EDA)

I then worked on understanding the data, such as visualizing distributions, correlation and key statistics. Then I identified key features to determine which features are most influential in predicting marathon performance.

Here is an example of my file: 

Open eda_feature_engineering.py

After running these scripts, a new data set was generated, which is as follows: 

Open engineered_marathon_data.csv

Step 4: Feature Engineering

I created additional features such as weekly mileage, average pace, heart rate zones, etc.

Here is an example of my file: 

Open model_training_evaluation.py

After running these scripts, the following output was created. I also added a saving functionality to the python script to save the model for steps to come down the line:

Step 5: Model Selection and Training

I choose models by selecting regression models like Linear Regression, Random Forest Regressor and so on. I then trained the models and split the data into training and test sets, to then train the models.

Here is an example of my file: 

Open model_selection_training.py

Below was the output upon running the script:

Linear Regression - Mean Squared Error (MSE): 0.3982072128259886
Random Forest - Mean Squared Error (MSE): 0.03546087955807133
Support Vector Machine - Mean Squared Error (MSE): 0.5529868195779899

Best Model - Random Forest with Grid Search - Mean Squared Error (MSE): 0.03800804209491479

Step 6: Model Evaluation 

I then evaluated the performance by using metrics like Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) to evaluate model performance. Using the model from step 4 and saving the information in the file. I created a script that will load the best model and then evaluate it. 

Here is an example of my file: 

Open model_evaluation.py

The following was my output from running the script, with the following graph: 

Mean Squared Error (MSE): 0.03992373710715418
Mean Absolute Error (MAE): 0.08257224627065758
R-squared (R2): 0.9743091986178533

Step 7: Creating a Training Program 

I then generated a personalized plans that uses the model to create training plans based on individual runner data. 

Here in an example of my file: 

Open generate_training_program.py

The following was my output after running the script:

Open training_programs.csv


Interpreting the data: 

The training_programs.csv file helps to use the model predictions in the context of a persons training needs, such as the Predicted Marathon Time and Training Program. For the predicted marathon time, each row corresponds to an individual’s predicted marathon time based on their specific training data (e.g. weekly mileage, speed, etc.). This prediction serves as an estimate of how long it might take them to complete a marathon under similar conditions. The training program column provides actionable advice based on the predictions. It will give you a beginner, intermediate and advanced route with different focuses depending on the level. For example at the beginner level, the program focuses on building endurance and gradually increasing mileage. At the intermediate level, it focuses on a balance of endurance and speed training and finally the advanced level focuses on high-intensity workouts and fine-tuning of the athletes current performance. 

Step 8: Deployment

The final step is to build a user interface such as create a web or mobile app for users to input their data and receive training plans. This step will be the next thing I start to look into to further build on my skills as a developer. 

