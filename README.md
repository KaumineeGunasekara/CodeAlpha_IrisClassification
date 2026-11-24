# CodeAlpha_IrisClassification

Task 1 of the CodeAlpha Data Science Internship: classify Iris flowers (setosa, versicolor, virginica) from sepal and petal measurements.

## Dataset
`Iris.csv` has 150 samples, 4 numeric features (sepal/petal length and width in cm) and 3 balanced classes (50 each). There are no missing values and 3 duplicate rows, which were kept because they are legitimate measurements.

## Approach
1. Load and inspect the data, and drop the `Id` column.
2. EDA: pairplot and correlation heatmap.
3. Stratified 80/20 train/test split (`random_state=42`).
4. Compare 5 models with 5-fold stratified cross-validation on the training set (scaling is inside a pipeline, so there is no leakage).
5. Evaluate the selected model on the held-out test set with accuracy, a classification report and a confusion matrix.

## Results
| Model | CV Acc | Test Acc |
|---|---|---|
| Logistic Regression | 0.958 | 0.933 |
| KNN (k=5) | 0.958 | 0.933 |
| **SVM (RBF)** | **0.967** | **0.967** |
| Decision Tree | 0.950 | 0.933 |
| Random Forest | 0.950 | 0.900 |

The SVM reached **96.7% test accuracy** (29 of 30 correct). Setosa is perfectly separable. The only error was one versicolor predicted as virginica.

With a test set of only 30 samples, one prediction moves accuracy by about 3.3 points, so the differences between models are within noise. The cross-validation scores are the more reliable comparison.

Petal length and petal width carry most of the predictive signal (about 87% of Random Forest importance combined).

## Run
```bash
pip install -r requirements.txt
python iris_classification.py
```

## Files
- `iris_classification.py` – full pipeline
- `model_comparison.csv` – model scores
- `images/` – plots
