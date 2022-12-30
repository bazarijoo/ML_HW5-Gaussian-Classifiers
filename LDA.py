from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from prepare_data import prepare_iris,prepare_vowel
from sklearn.model_selection import KFold
from preprocessing import preprocessing
import operator
import numpy as np


class LDA:

    def __init__(self,data):
        self.X = data.iloc[:,:-1].values
        # self.y=data.iloc[:,-1].values

        target_vals,target_counts=np.unique(data[data.columns[-1]],return_counts=True)
        self.phi = {target: pi_target for target, pi_target in zip(target_vals, target_counts / len(data))}
        self.mu_matrix ={target:target_mean for target,target_mean in zip(target_vals,np.array(data.groupby(data.columns[-1]).mean()))}   # last column
        self.sigma_matrix = data.iloc[:, :-1].cov().values  #except last column

    def LDA_score(self,class_label,test_data):
        sigma_inverse=np.linalg.pinv(self.sigma_matrix)
        mu_transpose=np.transpose(self.mu_matrix[class_label])

        return np.log(self.phi[class_label])-1/2*(np.linalg.multi_dot([self.mu_matrix[class_label],sigma_inverse,mu_transpose]))\
               +np.linalg.multi_dot([test_data,sigma_inverse,mu_transpose])

    def predict_class(self,test_data):
        X_test=test_data.iloc[:,:-1].values

        scores=[]
        for class_label in self.phi.keys():
            scores.append(self.LDA_score(class_label,X_test))

        all_target_scores=[{label:score for label,score in zip(self.phi.keys(),scores)} for scores in np.array(scores).T]# row wise score
        predicted_class=[]
        # finding maximum key of each row(data)
        for target_score in all_target_scores:
            predicted_class.append(max(target_score.items(), key=operator.itemgetter(1))[0])

        return predicted_class

def evaluate_guassian_classifier_crossvalidated(data):
    # train,test=train_test_split(data,test_size = 0.20, random_state = 5)
    # y_test = test.iloc[:, -1].values
    # model = LDA(train)
    # y_predicted = model.predict_class(test)
    # accuracy = len(y_test[y_predicted==y_test])/len(y_test)
    # print("Missclassification error is : ", "{0:.0%}".format(1-accuracy))
    # print("Accuracy is : ","{0:.0%}".format(accuracy))

    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    accuracies=[]

    for train_index, test_index in kf.split(data):
        train, test = data.iloc[train_index], data.iloc[test_index]
        y_test=test.iloc[:,-1].values
        model = LDA(train)
        y_predicted=model.predict_class(test)
        accuracies.append(len(y_test[y_predicted==y_test])/len(y_test))

    accuracy_mean=np.mean(accuracies)
    print("Cross Validation Missclassification error is : ", "{0:.0%}".format(1-accuracy_mean))
    print("Cross Validation Accuracy is : ","{0:.0%}".format(accuracy_mean))


def evaluate_guassian_classifier(train,test):
    model=LDA(train)
    y_predicted=model.predict_class(test)
    y_test=test.iloc[:,-1].values

    accuracy=len(y_test[y_test==y_predicted])/len(y_test)

    print("Missclassification error is : ", "{0:.0%}".format(1-accuracy))
    print("Accuracy is : ", "{0:.0%}".format(accuracy))


if __name__=='__main__':

    iris = prepare_iris()
    print("Testing classifier on Iris...")
    print("Testing LDA classifier wihout PCA or FDA :")
    evaluate_guassian_classifier_crossvalidated(iris)
    print()

    print("Testing classifier with PCA :")
    PCA_iris=preprocessing.PCA(iris)
    evaluate_guassian_classifier_crossvalidated(PCA_iris)
    print()

    print("Testing classifier with FDA")
    FDA_iris=preprocessing.FDA(iris)
    evaluate_guassian_classifier_crossvalidated(FDA_iris)
    print()


    vowel_train=prepare_vowel('vowel.train')
    vowel_test=prepare_vowel('vowel.test')
    print("Testing classifier on Vowel dataset...")
    print("Testing on vowel dataset without PCA or FDA :")
    evaluate_guassian_classifier(vowel_train,vowel_test)
    print()

    print("Testing on vowel with PCA ...")
    vowel_train_pca=preprocessing.PCA(vowel_train)
    vowel_test_pca=preprocessing.PCA(vowel_test)
    evaluate_guassian_classifier(vowel_train_pca, vowel_test_pca)
    print()

    print("Testing on vowel with FDA ...")
    vowel_train_fda=preprocessing.FDA(vowel_train)
    vowel_test_fda=preprocessing.FDA(vowel_test)
    evaluate_guassian_classifier(vowel_train_fda, vowel_test_fda)
    print()


