import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def test(predictions, true_labels):
    # Predictions: an array where each row is a prediction
    # 0 for dropouts, 1 for non-dropouts

    # Return Quality_Prediction, one row per prediction
    # Giving how many true zeros are classified and how many are not as well as
    # the average accuracy

    n = len(predictions)
    Quality_Prediction = np.zeros((3, n))

    for i in np.arange(n):
        success = predictions[i] == true_labels
        error = predictions[i] != true_labels
        neg = true_labels == 0
        pos = true_labels == 1

        tp = np.sum(success & pos)
        tn = np.sum(success & neg)
        fp = np.sum(error & pos)
        fn = np.sum(error & neg)

        # 0: TPR, 1:TNR, 2:Accuracy
        Quality_Prediction[0][i] = tp / (tp + fn)
        Quality_Prediction[1][i] = tn / (fp + tn)
        Quality_Prediction[2][i] = np.sum(success) / len(true_labels)

    return Quality_Prediction


def preprocess_em(data, labels, true_labels):
    nulls = data == 0
    labels = np.array(labels)
    true_labels = np.array(true_labels)

    labels_0 = labels[nulls]
    true_labels_0 = true_labels[nulls]

    labels_0 = np.array(labels_0).flatten()
    true_labels_0 = np.array(true_labels_0).flatten()

    return labels_0, true_labels_0


def roc(q, true_labels, data):
    thresholds = np.arange(0, 1, 0.05)
    tpr = np.zeros(len(thresholds))
    tnr = np.zeros(len(thresholds))
    acc = np.zeros(len(thresholds))

    for i, t in enumerate(thresholds):
        labels = np.zeros(q.shape)
        labels[q > t] = 1
        labels, true_labels0 = preprocess_em(data, labels, true_labels)
        tpr[i], tnr[i], acc[i] = test(np.array([labels]), true_labels0)

    plt.figure()
    plt.plot(tnr, tpr, marker='o')
    plt.xlabel('True Negative Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC analysis of the EM method')
    plt.xlim(1, 0)
    plt.show()


if __name__ == '__main__':
    q = pd.read_csv('../cache/q.csv', index_col=0)
    print(q.shape)
    # em_labels = pd.read_csv('../cache/ariane_labels.csv', index_col=0)
    true_labels = pd.read_csv('../cache/dropped_points.csv', index_col=0)
    data = pd.read_csv('../cache/synthetic_data.csv', index_col=0)

    # roc(q, true_labels, data)

    em_labels = np.zeros(q.shape)
    em_labels[q > 0.5] = 1

    em_labels, true_labels = preprocess_em(data, em_labels, true_labels)
    qp = test(np.array([em_labels]), true_labels)
    print(qp)
