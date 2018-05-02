import numpy as np
def test(predictions, true_labels):
    # Predictions: an array where each row is a prediction
    # 0 for dropouts, 1 for non-dropouts

    # Return Quality_Prediction, one row per prediction
    # Giving how many true zeros are classifed and how many are not as well as
    # the average accuracy

    n = predictions.shape[0]
    Quality_Prediction = np.zeros((3, n))

    for i in np.arrange(n):
        Quality_Prediction[1, i] = np.sum(predictions[i,:] == true_labels & true_labels == 0)
        Quality_Prediction[2, i] = np.sum(predictions[i,:] == true_labels & true_labels == 1)
        Quality_Prediction[3, i] = np.sum(predictions[i,:] == true_labels)/len(true_labels)

    return Quality_Prediction
