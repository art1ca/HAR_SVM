# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:26:25 2020

@author: WIN81
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def main():
    datos=pd.read_csv('wineq.csv')
    datos=datos.astype(float).fillna(0.0)
    
    y=datos.quality
    X=datos.drop('quality',axis=1)
    
    print(datos['quality'].value_counts())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y)
    
    sc = StandardScaler()
    X_train_array = sc.fit_transform(X_train.values)
    X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
    X_test_array = sc.transform(X_test.values)
    X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)
    
    clf=SVC(kernel='rbf').fit(X_train,y_train)
    print(clf.score(X_test,y_test))
    
if __name__=="__main__":
    main()
