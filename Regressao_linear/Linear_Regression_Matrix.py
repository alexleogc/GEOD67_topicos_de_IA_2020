import numpy as np

#======================================= REGRESSÃO LINEAR PSEUDO-INVERSA =================================#
def fit_linear_regression(X,y):
    """
    Entrada

    X - Conjunto de medidas
    y - Conjunto de observações

    Saída - coeficientes
    """
    X,y = np.asarray(X), np.asarray(y)
    return np.matmul(np.matmul(np.linalg.inv( np.matmul(X.T,X)),X.T),y)

#======================================= REGRESSÃO LINEAR PSEUDO-INVERSA =================================#

#======================================= PREDIÇÃO DA REGRESSÃO LINEAR =================================#

def predict_linear_regression(coef,X):
    """
    Entrada

    coef - coeficientes da regressão linear
    X    - Conjunto de medidas

    Saída - Y_predict (Predição)
    """
    return np.matmul(X,coef)

#======================================= PREDIÇÃO DA REGRESSÃO LINEAR =================================#

#======================================= REGRESSÃO LINEAR PSEUDO-INVERSA =================================#
def fit_linear_regression_ridge(X,y,alpha=0.0):
    """
    Entrada

    X - Conjunto de medidas
    y - Conjunto de observações

    Saída - coeficientes
    """
    X,y = np.asarray(X), np.asarray(y)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)+np.eye(N = np.matmul(X.T,X).shape[0])*alpha),X.T),y)
#======================================= REGRESSÃO LINEAR PSEUDO-INVERSA =================================#