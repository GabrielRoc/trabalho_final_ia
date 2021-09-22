import sys
import numpy as np
import cm as CmPrinter
import subprocess
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

def extract_features():
    print("\n\t------------  Features Extraction  ------------\n")
    subprocess.run(["python", "extract_pretrained_features.py", "--model", "vgg16", "--patches", "1", "--folds", "2", "--height", "500", "--width", "500", "-i", "Simpsons/%d/*.bmp", "-o", "Simpsons/features/fold-%d_patches-%d.npy", "--gpuid", "0"])

def read_data_ts():
    teste0_list = np.load("./Simpsons/features/fold-1_patches-1.npy")
    lista = [teste0_list]
    teste_list = []
    for i in range(len(teste0_list)):
        for j in lista:
            teste_list.append(j[i])

    return teste_list

def read_data_tr():
    treinamento0_list = np.load("./Simpsons/features/fold-0_patches-1.npy")
    lista = [treinamento0_list]
    treinamento_list = []
    for i in range(len(treinamento0_list)):
        for j in lista:
            treinamento_list.append(j[i])

    return treinamento_list

def define_class_ts():
    train_classes = []
    for j in range(78):
        train_classes.append(0)
    for j in range(61):
        train_classes.append(1)
    for j in range(33):
        train_classes.append(2)
    for j in range(30):
        train_classes.append(3)
    for j in range(24):
        train_classes.append(4)

    return train_classes

def define_class_tr():
    teste_classes = []
    for j in range(35):
        teste_classes.append(0)
    for j in range(25):
        teste_classes.append(1)
    for j in range(13):
        teste_classes.append(2)
    for j in range(12):
        teste_classes.append(3)
    for j in range(10):
        teste_classes.append(4)

    return teste_classes

def matriz_confusao(array, title):
    cmTitle = "Matriz de Confusão - " + title
    CmPrinter.plot_confusion_matrix(cm = array, target_names = ["Bart", "Homer", "Lisa", "Maggie", "Marge"], title = cmTitle, normalize = False, cmap = None)


## KNN Classificador
def knn(atrib_tr, classes_tr, atrib_ts, classes_ts):
    neigh = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
    neigh.fit(atrib_tr, classes_tr)
    print("\n\t-----------------  KNN  -----------------\n")
    print(classification_report(classes_ts, neigh.predict(atrib_ts)))
    print("Taxa de acerto: {}".format(accuracy_score(classes_ts, neigh.predict(atrib_ts))*100))
    matriz_confusao(confusion_matrix(classes_ts, neigh.predict(atrib_ts)), "KNN")
    return neigh.predict_proba(atrib_ts)

## MLP Classificador
def mlp(atrib_tr, classes_tr, atrib_ts, classes_ts):
    clf = MLPClassifier(solver='lbfgs')
    clf.fit(atrib_tr, classes_tr)
    print("\n\t-----------------  MLP  -----------------\n")
    print(classification_report((classes_ts), clf.predict(atrib_ts)))
    print("Taxa de acerto: {}".format(accuracy_score(classes_ts, clf.predict(atrib_ts))*100))
    matriz_confusao(confusion_matrix(classes_ts, clf.predict(atrib_ts)), "MLP")
    return clf.predict_proba(atrib_ts)

## RandomForest Classificador
def random_forest(atrib_tr, classes_tr, atrib_ts, classes_ts):
    clf = RandomForestClassifier()
    clf.fit(atrib_tr, classes_tr)
    print("\n\t------------  Random Forest  ------------\n")
    print(classification_report((classes_ts), clf.predict(atrib_ts)))
    print("Taxa de acerto: {}".format(accuracy_score(classes_ts, clf.predict(atrib_ts))*100))
    matriz_confusao(confusion_matrix(classes_ts, clf.predict(atrib_ts)), "Random Forest")
    return clf.predict_proba(atrib_ts)

# Sum Rule
def sum_rule(knn_result, mlp_result, rnd_result, classes_ts):
    vr = []
    for i in range(len(knn_result)):
        vr.append([])
        for j in range(len(knn_result[0])):
            vr[i].append(knn_result[i][j]+mlp_result[i][j]+rnd_result[i][j])
    for i in range(len(knn_result)):
        vr[i] = vr[i].index(max(vr[i]))
    print("\n\t-----------------  SUM  -----------------\n")
    print(classification_report(classes_ts, vr))
    print("Taxa de acerto: {}".format(accuracy_score(classes_ts, vr)*100))
    matriz_confusao(confusion_matrix(classes_ts, vr), "Sum Rule")
    return vr

def legenda():
    
    print("\n\t------------------  Legenda  ------------------\n")
    print("\t0: Bart\n")
    print("\t1: Homer\n")
    print("\t2: Lisa\n")
    print("\t3: Maggie\n")
    print("\t4: Marge\n")

if __name__ == "__main__":

    # Extração de caracteristicas
    extract_features()
    
    # Leitura dos arquivos
    tr = read_data_tr()
    ts = read_data_ts()

    # Definição das classes
    tr_class = define_class_tr()
    ts_class = define_class_ts()

    # Normalização Z-Score
    scaler = StandardScaler()
    trn = scaler.fit_transform(tr) 
    tsn = scaler.fit_transform(ts)

    knn_result = knn(tsn, tr_class, trn, ts_class)
    mlp_result = mlp(tsn, tr_class, trn, ts_class)
    rnd_result = random_forest(tsn, tr_class, trn, ts_class)

    sum_rule(knn_result, mlp_result, rnd_result, ts_class)

    legenda()