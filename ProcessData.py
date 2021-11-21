import os, sys
from IPython.display import display, Image
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn.decomposition import PCA

#hyperparameters
max_iters = [15, 50, 100]
num_components = [20, 30, 50, 75, 100, 150, 300, 400, 500]
alphas = [n / 7 for n in range(14)]
learning_rates = ['constant', 'invscaling', 'adaptive']
hiddenLayerSizes = [ (10), (20,), (30,), (30,10), (25,15), (50,10), (100,50), (200,300) ]

pca = PCA()

def get_labels(filename):
    lines = open(filename).readlines()[1:]
    labels = []
    for l in lines:
        labels.append(int(l.split(",")[1]))
    return labels

def show_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest', cmap='gray', vmin=0, vmax=255)
    plt.show()

def flatten(arr, size=784):
    new_arr = np.zeros( (len(arr), size) )
    for i in range(len(arr)):
        new_arr[i] = arr[i].flatten()
    return new_arr

def getScoreString(pred, true, maxiter, numcomponents, alpha, learning_rate, hiddenLayerSize, accuracy):
    output = ""
    output += "\n-------------- " + datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + " --------------\n"
    output += "\t\t" + "Accuracy: " + str(accuracy) + "%\n"
    output += "\n\t\t" + "Predicted: " + str(pred[:10])
    output += "\n\t\t" + "True: " + str(true[:10])
    output += "\n\t\t" + "PCA Num Components: " + str(numcomponents)
    output += "\n\t\t" + "Max Iter: " + str(maxiter)
    output += "\n\t\t" + "Alpha: " + str(alpha)
    output += "\n\t\t" + "Learning Rate: " + str(learning_rate)
    output += "\n\t\t" + "Hidden Layer Size: " + str(hiddenLayerSize)
    output += "\n"
    return output

def createSubmission(pred):
    submission = open("submission.txt", 'w')
    submission.write("ID,label\n")
    for i in range(len(pred)):
        submission.write(i + "," + pred[i] + "\n")
    submission.close()

def update_high_score(pred, true, maxiter, numcomponents, alpha, learning_rate, hiddenLayerSize, accuracy):
    try:
        data = open("best_result.txt", 'w')
        bestScore = float(data.readlines()[1][12:])
    except:
        data = [""]
        bestScore = 0

    if accuracy > bestScore:
        open("best_result.txt", 'w').write(getScoreString(pred, true, maxiter, numcomponents, alpha, learning_rate, hiddenLayerSize, accuracy)).close()
        createSubmission(pred)


def save_score(pred, true, maxiter, numcomponents, alpha, learning_rate, hiddenLayerSize, accuracy):
    correct = 0
    total = len(pred)
    for i in range(len(pred)):
        if int(pred[i]) == int(true[i]): correct += 1

    accuracy = correct/total*100
    update_high_score(pred, true, maxiter, numcomponents, alpha, learning_rate, hiddenLayerSize, accuracy)
    
    open("results.txt", 'w').write(getScoreString(pred, true, maxiter, numcomponents, alpha, learning_rate, hiddenLayerSize, accuracy)).close()


def run_model(X, y, testX, testY, maxiter, numcomponents, alpha, learning_rate, hiddenLayerSize):
    X = flatten(X)
    testX = flatten(testX)
    
    scalerx = preprocessing.StandardScaler().fit(X)
    X = scalerx.transform(X)
    testX = scalerx.transform(testX)

    pca.set_params(n_components = numcomponents)
    X = pca.fit_transform(X)
    testX = pca.transform(testX)

    model = MLPClassifier(random_state=1, max_iter=300, verbose = 10).fit(X, y)
    pred = model.predict(testX)
    save_score(pred, testY, maxiter, numcomponents, alpha, learning_rate, hiddenLayerSize)

##contrast = cv2.convertScaleAbs(train_images[150], alpha=alpha, beta=beta)
#nothing, contour = cv2.threshold(training_images_raw[150], 150, 255, cv2.THRESH_BINARY_INV)
#show_image(contrast)

#load in data
all_images = np.load('train_images.npy')
all_labels = get_labels('train_labels.csv')

labels = { 0:"T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat", 5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot" }

train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels)

for m in max_iters:
    for nc in num_components:
        for a in alphas:
            for l in learning_rates:
                for h in hiddenLayerSizes:
                    run_model(train_images, train_labels, test_images, test_labels, m, nc, a, l, h )


#next:
#standardize variables
#add a couple more models
#train and tune baby


#test_images_raw = np.load('test_images.npy')