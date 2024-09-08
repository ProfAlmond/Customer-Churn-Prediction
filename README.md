# Customer Churn Prediction Project

## Overview

This project focuses on predicting customer churn in the telecom industry. Customer churn is a significant issue where customers leave one service provider for another, costing companies a lot in terms of customer acquisition and retention. The aim of this project is to develop a predictive model that identifies customers likely to churn, allowing the telecom company to take proactive measures to retain them.

I explored and compared several machine learning models, including XGBoost, Keras Neural Network, Simple RNN, and LSTM, to identify the best-performing one. After a thorough analysis, I selected the most suitable model based on evaluation metrics like accuracy, precision, recall, F1-score, and ROC-AUC.

## Dataset

The dataset used in this project is the IBM Sample Data Set for customer churn. It contains customer attributes such as:

- **Churn Status**: Indicates if the customer left in the last month.
- **Services Signed Up**: Includes phone service, multiple lines, internet service, online security, online backup, device protection, tech support, streaming TV, and streaming movies.
- **Account Information**: Tenure, contract type, payment method, paperless billing, monthly charges, and total charges.
- **Demographics**: Gender, age range, partner status, and dependents.

The dataset allowed me to analyze patterns and factors contributing to customer churn.

## Objective

The objective of this project was to build a predictive model capable of accurately identifying customers who are likely to churn. By doing so, telecom companies can develop focused customer retention programs and improve their service quality. The project involved data preprocessing, model building, and performance evaluation to find the best model for churn prediction.

## Models and Evaluation Metrics

I evaluated the following models using various performance metrics:

1. **XGBoost Classifier**
   - Accuracy: 0.84
   - Precision: 0.85 (Class 0), 0.83 (Class 1)
   - Recall: 0.83 (Class 0), 0.85 (Class 1)
   - F1-Score: 0.84 (Both classes)
   - ROC-AUC: 0.8761

2. **Keras Neural Network**
   - Accuracy: 0.79
   - Precision: 0.84 (Class 0), 0.76 (Class 1)
   - Recall: 0.73 (Class 0), 0.86 (Class 1)
   - F1-Score: 0.78 (Class 0), 0.81 (Class 1)
   - ROC-AUC: 0.8761

3. **Simple RNN**
   - Accuracy: 0.81
   - Precision: 0.79 (Class 0), 0.84 (Class 1)
   - Recall: 0.86 (Class 0), 0.77 (Class 1)
   - F1-Score: 0.82 (Class 0), 0.80 (Class 1)
   - ROC-AUC: 0.8997

4. **LSTM** (Selected Model)
   - Accuracy: 0.84
   - Precision: 0.83 (Both classes)
   - Recall: 0.84 (Class 0), 0.83 (Class 1)
   - F1-Score: 0.84 (Class 0), 0.83 (Class 1)
   - ROC-AUC: 0.9241

### Selected Model: LSTM
After comparing all models, the LSTM model stood out as the best performer. It had the highest ROC-AUC score (0.9241) and demonstrated strong predictive capabilities for customer churn, making it the most suitable choice for this project.

## Conclusion and Future Work

This project successfully developed a predictive model to help telecom companies identify potential customer churners. The LSTM model was chosen due to its balanced performance and high accuracy. 

### Future Work
- **Further Feature Engineering**: Creating new features and refining existing ones could further improve model performance.
- **Hyperparameter Tuning**: Further tuning the hyperparameters of the selected models might yield even better results.
- **Model Deployment**: The next step is to deploy the model for real-time predictions and integrate it into a customer retention strategy.

## Project Files

- **Customer_Churn_Dataset.csv**: The dataset used for analysis.
- **Churn_Prediction_Model.ipynb**: The Jupyter notebook containing the code for data exploration, preprocessing, model building, and evaluation.
- **README.md**: This readme file that explains the project.

## How to Use This Project

1. **Clone the Repository**: Clone this repository to your local machine.
2. **Install Dependencies**: Ensure that you have all necessary Python libraries installed, such as Pandas, NumPy, Scikit-learn, TensorFlow, and XGBoost.
3. **Run the Notebook**: Open the `Churn_Prediction_Model.ipynb` notebook and run the code to reproduce the results.
4. **Evaluate the Model**: Analyze the model's performance and adjust as needed based on the project requirements.

## Contact Information

For any questions or further discussion regarding this project, feel free to reach out:

- **Email**: alimtulufunda@gmail.com
- **LinkedIn**: [Ali Mtulu](http://www.linkedin.com/in/ali-mtulu)
