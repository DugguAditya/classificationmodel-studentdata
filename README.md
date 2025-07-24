# Student Placement Prediction Model

## Project Overview
This project implements an end-to-end machine learning pipeline to predict student placement outcomes based on CGPA and IQ scores. The model uses logistic regression to classify whether a student will get placed or not.

## Dataset
- **File**: `placement.csv`
- **Features**: 
  - CGPA (Continuous variable)
  - IQ (Continuous variable) 
  - Placement (Target variable - binary classification)
- **Size**: 100 students

## Machine Learning Pipeline

### 1. Data Loading & Exploration
- Load data using pandas
- Explore dataset structure with `df.info()`
- Remove unnecessary columns (first column dropped)
- Visualize data distribution using scatter plots

### 2. Data Preprocessing
- **Feature Selection**: CGPA and IQ as input features (X)
- **Target Variable**: Placement status (y)
- **Train-Test Split**: 90% training, 10% testing
- **Feature Scaling**: StandardScaler for normalization

### 3. Model Training
- **Algorithm**: Logistic Regression
- **Library**: scikit-learn
- **Features**: Standardized CGPA and IQ scores

### 4. Model Evaluation
- **Metric**: Accuracy Score
- **Visualization**: Decision boundary plotting using mlxtend

### 5. Model Persistence
- **Format**: Pickle file (`Model.pkl`)
- **Usage**: Can be loaded for future predictions

## Technical Stack
- **Python Libraries**:
  - `pandas` - Data manipulation
  - `numpy` - Numerical computations
  - `matplotlib` - Basic plotting
  - `scikit-learn` - Machine learning algorithms
  - `mlxtend` - Advanced ML visualizations
  - `pickle` - Model serialization

## Installation & Setup
```bash
pip install pandas numpy matplotlib scikit-learn mlxtend
```

## How to Run
1. Open `end-to-end-ml.ipynb` in Jupyter Notebook/Lab
2. Run cells sequentially from top to bottom
3. The trained model will be saved as `Model.pkl`

## Workflow Steps
1. **Import Libraries** - Load required Python packages
2. **Load Data** - Read placement.csv file
3. **Data Exploration** - Examine dataset structure and info
4. **Data Cleaning** - Remove unnecessary columns
5. **Visualization** - Create scatter plot of features vs placement
6. **Feature Engineering** - Separate features (X) and target (y)
7. **Data Splitting** - Create train/test sets
8. **Feature Scaling** - Standardize input features
9. **Model Training** - Fit logistic regression model
10. **Prediction** - Generate predictions on test set
11. **Evaluation** - Calculate accuracy score
12. **Visualization** - Plot decision regions
13. **Model Saving** - Serialize model using pickle

## File Structure
```
├── end-to-end-ml.ipynb    # Main notebook with complete ML pipeline
├── placement.csv          # Student dataset
├── Model.pkl             # Trained logistic regression model
├── README.md             # This documentation
└── ML-Practice/          # Additional practice files
    ├── practice.ipynb
    ├── train.json
    └── world.sql
```

## Key Features
- ✅ Complete end-to-end ML pipeline
- ✅ Data visualization and exploration
- ✅ Feature scaling and preprocessing
- ✅ Model training and evaluation
- ✅ Decision boundary visualization
- ✅ Model persistence for deployment

## Results
The model predicts student placement based on CGPA and IQ with measurable accuracy. Decision regions are visualized to understand the model's classification boundaries.

## Usage Example
```python
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = pickle.load(open('Model.pkl', 'rb'))

# Prepare new data (CGPA, IQ)
new_data = [[8.5, 110]]  # Example: CGPA=8.5, IQ=110

# Scale the data (use same scaler as training)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(new_data)

# Make prediction
prediction = model.predict(scaled_data)
print(f"Placement prediction: {'Placed' if prediction[0] == 1 else 'Not Placed'}")
```

## Author
Aditya

## License
This project is for educational purposes.
