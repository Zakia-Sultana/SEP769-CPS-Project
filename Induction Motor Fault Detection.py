import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def load_data(data_folder):
    data = []
    labels = []
    for folder in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder)
        if folder == 'normal':
            label = 0
        elif folder == 'imbalance':
            label = 1
        else:
            continue

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, header=None)
            if df.shape[1] == 8:
                data.extend(df.values)
                labels.extend([label] * df.shape[0])
            else:
                print(f"Invalid shape found in {file_path}. Skipping...")

    data = np.array(data)
    labels = np.array(labels)
    return data, labels

def Preprocess_data(data):
    # To replace the null values we will use this part
    imputer = SimpleImputer(strategy='mean')
    data = imputer.fit_transform(data)

    return data
def train_model(data, labels):
    data = Preprocess_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    #We will use random forest classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    #Prediction using test data
    y_pred = classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Get ROC curve data for plotting accuracy
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Get confusion matrix for plotting accuracy
    cm = confusion_matrix(y_test, y_pred)

    return classifier, fpr, tpr, roc_auc, cm

def plot_roc_curve(fpr, tpr, roc_auc):
#Google: An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def Confusion_matrix_plotting(cm):
#Google: A confusion matrix presents a table layout of the different outcomes of the prediction and results of a classification problem and helps visualize its outcomes
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Normal', 'Imbalance']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def plot_precision_recall_f1(precision, recall, f1):
#Google: F1 score is a machine learning evaluation metric that measures a model's accuracy. It combines the precision and recall scores of a model.
# The accuracy metric computes how many times a model made a correct prediction across the entire dataset
    plt.figure()
    classes = ['Normal', 'Imbalance']
    x = np.arange(len(classes))
    width = 0.2
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-Score')
    plt.xticks(x, classes)
    plt.ylabel('Scores')
    plt.title('Precision, Recall, and F1-Score Comparison')
    plt.legend(loc="lower right")
    plt.show()

def main():
    data_folder = 'C:/Users/zakia/Downloads/CPS_project_data/test'
    data, labels = load_data(data_folder)


    classifier, fpr, tpr, roc_auc, cm = train_model(data, labels)

    # Plot ROC curve and confusion matrix
    plot_roc_curve(fpr, tpr, roc_auc)
    Confusion_matrix_plotting(cm)

    # Test prediction with sample sensor values
    sample_sensor_values = [ -0.8, -1.0, -0.7, -0.2, 0.0, 0.0, 0.3, 0.1]
    prediction = classifier.predict([sample_sensor_values])

    if prediction[0] == 0:
        print("The motor is normal.")
    else:
        print("The motor is imbalanced and may be experiencing failure.")


    y_test_predic = classifier.predict(data)
    class_report = classification_report(labels, y_test_predic, target_names=['Normal', 'Imbalance'], output_dict=True)
#Calling the plots based on tested data
    precision = [class_report['Normal']['precision'], class_report['Imbalance']['precision']]
    recall = [class_report['Normal']['recall'], class_report['Imbalance']['recall']]
    f1 = [class_report['Normal']['f1-score'], class_report['Imbalance']['f1-score']]
    plot_precision_recall_f1(precision, recall, f1)

if __name__ == "__main__":
    main()
