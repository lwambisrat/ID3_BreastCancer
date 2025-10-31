# ID3 Breast Cancer Decision Tree Classifier

This project loads a breast cancer dataset, cleans the data, and builds a decision tree classifier using the ID3 algorithm. It performs k-fold cross-validation to evaluate the model’s accuracy, providing insights into its predictive performance on unseen data.

---

## Table of Contents

- Features
- Requirements and Setup instructions
- Usage
- Project structure

---

## Features

- Data cleaning: it replaces missing values and removes that row 
- Entropy and information gain calculations for feature selection
- Recursive ID3 decision tree builder with stopping criteria:
  - All samples in node belong to one class
  - No features remain or node has fewer than minimum samples
- Prediction function traversing the constructed tree
- 10-fold cross-validation for model evaluation
- Prints accuracy for each fold and average accuracy after all folds

---

## Requirements and Setup Instructions

1. Clone the repository

   Using SSH:

    - git clone git@github.com:lwambisrat/ID3_BreastCancer.git
    - cd ID3_BreastCancer

   Or using HTTPS:
    - git clone https://github.com/lwambisrat/ID3_BreastCancer.git
    - cd ID3_BreastCancer
   
2. Python 3.x or above  

3. Create a virtual environment (recommended):
   
   `uv venv venv`

4. Activate the virtual environment:

  - On Linux/macOS:
    
    `source venv/bin/activate`
    
  - On Windows (Command Prompt):
    
    `venv\Scripts\activate`

3. Install the required Python packages:
   
   - `uv pip install pandas numpy matplotlib`
   
   - `uv pip install ipykernel -U --force-reinstall`

---

## Usage

 #### Option 1: 
 - Run the classifier script:
  
     `python id3_breast_cancer.py `   
 #### Option 2:
- Open the file `id3_breast_cancer.ipynb` in VS Code.  
- Select the Python environment/kernel you created.  
- Click **Run All** or run each cell step-by-step.  

The script will:

- Load and clean the breast cancer dataset from `data/breast-cancer.data`
- Perform 10-fold cross-validation
- Print accuracy for each fold and the average accuracy

---

## Project structure

ID3_BreastCancer/

- data/

   - breast-cancer.data   
   
   - breast-cancer.names    

- id3_breast_cancer.py   

- id3_breast_cancer.ipynb    

- README.md 


---

## Tools & Technologies Used

- **Python 3.10.12** — main programming language  
- **pandas** — for data loading, cleaning, and manipulation  
- **NumPy** — for numerical operations  
- **Matplotlib** — for visualization in the `.ipynb` notebook  
- **VS Code Notebook kernel** — to execute and inspect data interactively  
- **Built-in math & random modules** — for entropy, log₂, and shuffling  
- **Manual ID3 implementation** — no external ML libraries used  
- **K-fold cross-validation** — to evaluate model performance  
- **Virtual environment (uv venv)** — to isolate dependencies

























