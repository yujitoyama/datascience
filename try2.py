import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import tree
from sklearn import metrics

def _43():
    digits = datasets.load_digits()
    for label,img in zip(digits.target[:10],digits.images[:10]):
        plt.subplot(2,5,label+1)
        plt.axis("off")
        plt.imshow(img,cmap=plt.cm.gray_r,interpolation='nearest')
    
    plt.show()

def bunrui1():
    digits = datasets.load_digits()
    flag_3_8 = (digits.target==3) + (digits.target == 8)
    images = digits.images[flag_3_8]
    labels = digits.target[flag_3_8]
    #3次元画像の一元化
    images = images.reshape(images.shape[0],-1)
    #分類器の生成
    n_sample = len(flag_3_8[flag_3_8])
    train_size = int(n_sample*3/5)
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(images[:train_size],labels[:train_size])
    #評価
    expected = labels[train_size:]
    predicted = classifier.predict(images[train_size:])
    print("Accuracy:\n",metrics.accuracy_score(expected,predicted))
    print("Confusion matrix:\n",metrics.confusion_matrix(expected,predicted))
    print("Precision:\n",metrics.precision_score(expected,predicted,pos_label=3))
    print("Recall:\n",metrics.recall_score(expected,predicted,pos_label=3))
    print("F-measure:\n",metrics.f1_score(expected,predicted,pos_label=3))
    
if __name__ == "__main__":
    bunrui1()
