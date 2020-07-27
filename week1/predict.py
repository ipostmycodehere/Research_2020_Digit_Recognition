def predict(X_test,X_train,Y_train):
    Y_predict = np.zeros((len(X_test),))
    for i in range(len(X_test)):
        d = np.sum(abs(X_test[i] -  X_train),axis=1)
        ind = np.argmin(d)
        Y_predict[i] = Y_train[ind]
    return Y_predict

Y_predict = predict(X_test, X_train, Y_train)
