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
max_iters = [250, 500]
num_components = [50, 100, 300, 500]
alphas = [n / 7 for n in range(7)]
learning_rates = ['constant', 'invscaling', 'adaptive']
hiddenLayerSizes = [ (20,), (30,), (100,), (30,10), (25,15), (50,10), (100,50), (200,300), (50,30) ]

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
    output += "\n\t\t" + "Accuracy: " + str(accuracy) + "%\n"
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
    submission = open("submission.csv", 'w')
    submission.write("ID,label\n")
    for i in range(len(pred)):
        if i != len(pred)-1: submission.write(str(i) + "," + str(pred[i]) + "\n")
        else: submission.write(str(i) + "," + str(pred[i]))
    submission.close()

def update_high_score(pred, true, maxiter, numcomponents, alpha, learning_rate, hiddenLayerSize, accuracy):
    try:
        data = open("best_result.txt", 'r')
        bestScore = float(data.readlines()[3][12:16])
    except:
        bestScore = 0

    data.close()

    print(bestScore)

    if accuracy > bestScore:
        file = open("best_result.txt", 'w')
        file.write(getScoreString(pred, true, maxiter, numcomponents, alpha, learning_rate, hiddenLayerSize, accuracy))
        file.close()
        createSubmission(pred)


def save_score(pred, true, maxiter, numcomponents, alpha, learning_rate, hiddenLayerSize):
    correct = 0
    total = len(pred)
    for i in range(len(pred)):
        if int(pred[i]) == int(true[i]): correct += 1

    accuracy = correct/total*100
    update_high_score(pred, true, maxiter, numcomponents, alpha, learning_rate, hiddenLayerSize, accuracy)
    
    open("results.txt", 'a').write(getScoreString(pred, true, maxiter, numcomponents, alpha, learning_rate, hiddenLayerSize, accuracy))


def run_model(X, y, testX, testY, maxiter, numcomponents, alpha, learning_rate, hiddenLayerSize):

    X = flatten(X)
    scalerx = preprocessing.StandardScaler().fit(X)
    X = scalerx.transform(X)
    pca.set_params(n_components = numcomponents)
    X = pca.fit_transform(X)

    model = MLPClassifier(alpha=alpha, learning_rate=learning_rate, hidden_layer_sizes=hiddenLayerSize, random_state=1, max_iter=maxiter, verbose = 1, tol=0.00001).fit(X, y)

    if (testX != None and testY != None):
        testX = flatten(testX)
        testX = scalerx.transform(testX)
        testX = pca.transform(testX)
        pred = model.predict(testX)
        save_score(pred, testY, maxiter, numcomponents, alpha, learning_rate, hiddenLayerSize)

    return model, scalerx, pca

##contrast = cv2.convertScaleAbs(train_images[150], alpha=alpha, beta=beta)
#nothing, contour = cv2.threshold(training_images_raw[150], 150, 255, cv2.THRESH_BINARY_INV)
#show_image(contrast)

#load in data

labels = { 0:"T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat", 5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot" }

#train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels)



#for m in max_iters:
#    for nc in num_components:
#        for a in alphas:
#            for l in learning_rates:
#                for h in hiddenLayerSizes:
#                    run_model(train_images, train_labels, test_images, test_labels, m, nc, a, l, h )


def createFinalModel(maxiter, numc, alpha, learnrate, hiddenLayers):
    

    all_images = np.load('train_images.npy')
    all_labels = get_labels('train_labels.csv')

    finalModel, scalerx, pca = run_model(all_images, all_labels, None, None, maxiter, numc, alpha, learnrate, hiddenLayers )
    
    final_test_images = flatten(np.load('test_images.npy'))
    final_test_images = scalerx.transform(final_test_images)
    final_test_images = pca.transform(final_test_images)

    finalPred = finalModel.predict(final_test_images)
    createSubmission(finalPred)


createFinalModel(1000, 100, 5.0/7.0, 'constant', (100,))