Heart Disease Prediction – Machine Learning Project

Project Overview
My project aims to predict heart disease using multiple machine-learning algorithms. Each algorithm is implemented in a separate Jupyter notebook so I can compare their performance. The project uses the Heart Cleveland dataset and covers data preprocessing, model training, performance evaluation, and accuracy comparison across all models. This project demonstrates how machine learning can support early diagnosis and assist healthcare decision-making.

Project Structure
heart disease prediction/
- adaboost.ipynb
- artificial neural network.ipynb
- decision treee.ipynb
- gradient boosting.ipynb
- heart_cleveland_upload-checkpoint.csv
- knn.ipynb
- linear regression.ipynb
- logistic regression.ipynb
- naive bayes.ipynb
- random forest.ipynb
- support vector machine(svm).ipynb

Machine Learning Models Used
1. Logistic Regression – baseline binary classifier.
2. Support Vector Machine (SVM) – hyperplane and kernel-based classifier.
3. Random Forest – ensemble of decision trees.
4. Gradient Boosting – sequential boosting model.
5. AdaBoost – converts weak learners into strong ones.
6. KNN – distance-based prediction.
7. Decision Tree – interpretable rule-based classifier.
8. Naive Bayes – probabilistic classifier.
9. Linear Regression – used for analysis.
10. Artificial Neural Network (ANN) – deep learning model.

Dataset Information
Includes features such as age, sex, chest pain type, BP, cholesterol, ECG results, heart rate, angina, oldpeak, major vessels, and thalassemia.
Target: 0 – No Heart Disease, 1 – Heart Disease.

Steps Performed
- Data loading and cleaning
- Exploratory analysis
- Feature engineering
- Scaling and normalization
- Train–test split
- Training 11 ML models
- Evaluating accuracy, precision, recall, F1-score
- Confusion matrix analysis
- Model comparison

How to Run
pip install numpy pandas matplotlib scikit-learn
jupyter notebook
Open any model file and run all cells.

Results Summary
Ensemble models (RF, GB, AdaBoost) perform strongly.  
SVM works well after scaling.  
ANN performs effectively with tuning.  
Logistic Regression & KNN used as baselines.

Future Enhancements
- Hyperparameter tuning
- Deep-learning models
- Deployment via Streamlit/Flask
- Feature importance visualizations
- Cross-validation

Conclusion
This project shows how machine-learning algorithms can predict heart disease. Evaluating 11 models highlights their strengths and applicability in real-world medical prediction.
