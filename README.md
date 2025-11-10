 # ID3 Breast Cancer Decision Tree Classifier

This project loads a breast cancer dataset, cleans the data, and builds a decision tree classifier using the ID3 algorithm. It performs k-fold cross-validation to evaluate the model’s accuracy, providing insights into its predictive performance on unseen data.

---

## Table of Contents

- Features
- Requirements
- Project structure
- Setup instructions
- Usage


---

## Features

- Cleans and preprocesses data (handles missing values and drops incomplete rows)
- Calculates **entropy** and **information gain** for feature selection
- Recursively builds a **Decision Tree** using the ID3 algorithm
- Stopping conditions:
  - All samples in a node belong to one class
  - No remaining features
  - Node has fewer than the minimum required samples
- Performs **10-fold cross-validation**
- Prints accuracy for each fold and the average accuracy
- Includes a **.ipynb notebook file** (`id3_breast_cancer.ipynb`) that can be run directly inside **VS Code**

---

## Requirements

- Dataset: [Breast Cancer Data – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/14/breast+cancer)
- Python 3.x or above
- Clone the repository


   Using SSH:

    - git clone git@github.com:lwambisrat/ID3_BreastCancer.git
    - cd ID3_BreastCancer

   Or using HTTPS:
    - git clone https://github.com/lwambisrat/ID3_BreastCancer.git
    - cd ID3_BreastCancer
      
---

## Project Structure

ID3_BreastCancer/

├── data/

     ├── breast-cancer.data        # Breast cancer dataset file
     └── breast-cancer.names       # Dataset description 

├── id3_breast_cancer.py          

├── id3_breast_cancer.ipynb       

├── README.md                     


---



## Setup Instructions

1. Create a virtual environment:
   
       uv venv venv 

3. Activate the virtual environment:

  - On Linux/macOS:
    
        source venv/bin/activate
    
  - On Windows (Command Prompt):
    
        venv\Scripts\activate

3. Install the required Python packages
 
        uv pip install pandas numpy matplotlib
  
        uv pip install ipykernel -U --force-reinstall

---

## Usage

 #### Option 1: 
 - Run the classifier script:
  
       python id3_breast_cancer.py   
 #### Option 2:
- Open the file `id3_breast_cancer.ipynb` in VS Code.  
- Select the Python environment/kernel you created.  
- Click **Run All** or run each cell step-by-step.  

The script will:

- Load and clean the breast cancer dataset from `data/breast-cancer.data`
- Perform 10-fold cross-validation
- Print accuracy for each fold and the average accuracy
- The out put is going to be same format with below logs
  
  `Loaded 277 samples after cleaning.` 

  `Fold 1 Accuracy: 0.556`

  `Fold 2 Accuracy: 0.667`

  `Fold 3 Accuracy: 0.704`

  `Fold 4 Accuracy: 0.556`

  `Fold 5 Accuracy: 0.593`

  `Fold 6 Accuracy: 0.556`

  `Fold 7 Accuracy: 0.667`

  `Fold 8 Accuracy: 0.593`

  `Fold 9 Accuracy: 0.593`

  `Fold 10 Accuracy: 0.704`

  `Average Accuracy: 0.611` 

 














