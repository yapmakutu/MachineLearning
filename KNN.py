import cv2
import numpy as np
from skimage import feature
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Function: Extract LBP features from an image
def extract_lbp_features(image):
    #feature.local_binary_pattern: This is a function provided by the scikit-image
    #library that computes the local binary pattern of an image.
    #P defines the number of points in the circular neighborhood of each pixel.
    #R defines the radius of the circle for the neighborhood
    #method:This parameter specifies the method used to calculate the LBP.
    lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    return hist
#Labeling:
#The get_label_from_path function extracts labels based on the presence of
#keywords ('malignant', 'benign', 'normal') in the image filenames.
def get_label_from_path(image_path):
    filename = os.path.basename(image_path)

    if 'malignant' in filename:
        return 0  # 0 represents malignant
    elif 'benign' in filename:
        return 1  # 1 represents benign
    elif 'normal' in filename:
        return 2  # 2 represents normal
    else:
        return None

path_to_data = r"C:\Users\Lenovo\Desktop\BreastCancerDataset\deneme"

#Data Conversion:
#Feature vectors (feature_list) and labels (label_list) are collected for each image in the dataset.
feature_list = []  # Features
label_list = []    # Labels

# Iterate through each subdirectory (malignant, benign, normal)
for subdir in ['malignant', 'benign', 'normal']:
    subdir_path = os.path.join(path_to_data, subdir)
    if os.path.isdir(subdir_path):
        # Iterate through each image file in the subdirectory
        for image_file in os.listdir(subdir_path):
            if image_file.endswith('.jpg') or image_file.endswith('.png'):
                image_path = os.path.join(subdir_path, image_file)

                # Load the image using OpenCV
                original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if original_image is None:
                    print(f"Failed to load image: {image_path}")
                    continue
                lbp_features = extract_lbp_features(original_image)

                feature_list.append(lbp_features)

                # Get label from image path
                label = get_label_from_path(image_path)
                label_list.append(label)

# Convert lists to NumPy arrays
#X is typically used to represent the feature matrix, where each row
#corresponds to a data point (e.g., an image), and each column corresponds to a feature (LBP histogram bins).
#y is used to represent the labels corresponding to each data point in X.
# Convert lists to NumPy arrays
X = np.array(feature_list)
y = np.array(label_list)


#Data Splitting:
#The dataset is split into training and testing sets using the train_test_split function.
# 80% of the data is used for training (X_train, y_train), and 20% is used for testing (X_test, y_test).
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# The random split helps ensure that the model is tested on data it hasn't seen during training, and the
#random_state parameter makes the split reproducible.

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grid for grid search
param_grid = {'n_neighbors': [3, 5, 7, 10]}

# Initialize KNN classifier
knn_classifier = KNeighborsClassifier()

# Use GridSearchCV for hyperparameter tuning
#A grid search (GridSearchCV) is employed to find the best hyperparameter (n_neighbors) by trying different values.
#A K-Nearest Neighbors (KNN) classifier is trained with hyperparameter tuning performed through a grid search.
grid_search = GridSearchCV(knn_classifier, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Use the best model for prediction
best_knn_classifier = grid_search.best_estimator_
y_pred = best_knn_classifier.predict(X_test_scaled)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Best Parameters: {best_params}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')



# Function: Extract LBP features from an image
def extract_lbp_features(image):
    lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    return hist

# Function: Get label from image path (customize this based on your dataset)
def get_label_from_path(image_path):
    filename = os.path.basename(image_path)

    if 'malignant' in filename:
        return 0  # 0 represents malignant
    elif 'benign' in filename:
        return 1  # 1 represents benign
    elif 'normal' in filename:
        return 2  # 2 represents normal
    else:
        return None

# Specify the path to the image you want to test
user_input_path = input("Enter the path to the image you want to classify: ")

# Load the user input image using OpenCV
user_input_image = cv2.imread(user_input_path, cv2.IMREAD_GRAYSCALE)

if user_input_image is None:
    print(f"Failed to load the input image: {user_input_path}")
else:
    # Extract LBP features from the user input image
    user_input_lbp_features = extract_lbp_features(user_input_image)

    # Scale the features using the same scaler used in training
    user_input_features_scaled = scaler.transform([user_input_lbp_features])

    # Use the trained KNN classifier to predict the label
    predicted_label = best_knn_classifier.predict(user_input_features_scaled)[0]

    # Get the human-readable label
    label_mapping = {0: 'Malignant', 1: 'Benign', 2: 'Normal'}
    predicted_label_human_readable = label_mapping[predicted_label]

    print(f"The predicted label for the input image is: {predicted_label_human_readable}")
