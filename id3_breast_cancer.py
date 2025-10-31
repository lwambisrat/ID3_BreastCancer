import pandas as pd
import numpy as np
from math import log2
from random import shuffle

DATA_PATH = "data/breast-cancer.data"  # File path of the dataset
MIN_SAMPLES_SPLIT = 5  # Minimum samples required to split a node (K)
NUM_FOLDS = 10   # Number of folds for cross-validation

# 1. Load and clean data(remove empty row and replace missing values("?") with NAN)
def load_data(path):
    column_names = [
        "class", "age", "menopause", "tumor-size", "inv-nodes",
        "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"
    ]
    df = pd.read_csv(path, header=None, names=column_names)
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    return df.reset_index(drop=True)

# 2.Calculate the Entropy of the dataset and Compute the Information Gain for the given attribute
def entropy(data):
    labels = data['class'].value_counts(normalize=True)
    return -sum(p * log2(p) for p in labels)

def information_gain(data, feature):
    total_entropy = entropy(data)
    values = data[feature].unique()
    weighted_entropy = 0
    for v in values:
        subset = data[data[feature] == v]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)
    return total_entropy - weighted_entropy

# 3. Build ID3 Tree recursively
# - Stops if all samples belong to one class.
# - Stops if no features remain or data size < MIN_SAMPLES_SPLIT.
# - Selects the best feature based on maximum information gain.
# - Builds subtrees for each feature value.
def build_tree(data, features):
    if len(data['class'].unique()) == 1:
        return data['class'].iloc[0]

    if len(features) == 0 or len(data) < MIN_SAMPLES_SPLIT:
        return data['class'].mode()[0]

    gains = {f: information_gain(data, f) for f in features}
    best_feature = max(gains, key=gains.get)

    tree = {best_feature: {}}
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        if subset.empty:
            tree[best_feature][value] = data['class'].mode()[0]
        else:
            new_features = [f for f in features if f != best_feature]
            tree[best_feature][value] = build_tree(subset, new_features)
    return tree

# 4.Predicts the class label for a single sample by traversing the decision tree.
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    value = sample[feature]
    if value in tree[feature]:
        return predict(tree[feature][value], sample)
    else:
        return "unknown"  

# 5. K-FOLD Cross Validation to Randomly shuffles the dataset,splits data into K folds,trains on K-1 folds and tests on the remaining fold, and repeats the process K times and reports accuracy for each.

def k_fold_cross_validation(df, k=NUM_FOLDS):
    data = df.sample(frac=1).reset_index(drop=True) 
    fold_size = len(data) // k
    accuracies = []

    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        test = data.iloc[start:end]
        train = pd.concat([data.iloc[:start], data.iloc[end:]])

        features = [f for f in train.columns if f != 'class']
        tree = build_tree(train, features)

        correct = 0
        for _, row in test.iterrows():
            pred = predict(tree, row)
            if pred == row['class']:
                correct += 1
        accuracy = correct / len(test)
        accuracies.append(accuracy)
        print(f"Fold {i+1} Accuracy: {accuracy:.3f}")

    print("\nAverage Accuracy:", np.mean(accuracies).round(3))
    return accuracies

# 6. Main execution that coordinates the full workflow
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} samples after cleaning.\n")
    accuracies = k_fold_cross_validation(df)
