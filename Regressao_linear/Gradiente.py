import numpy as np

def predict(coef,X):
    """
    Calculo matricial de uma função linear:
    
    X - matriz das observações; e
    coef - Variáveis da função calculada
    """
    
    X = np.asarray(X) ; coef = np.asarray(coef)
    return X.dot(coef)

def cost(theta,X,y):
    """
    Função custo que computa os desvios entre o dado observado e o calculado:
    X - Matriz das variáveis observadas (nxm : n - número de variáveis ; m - número de observações)
    y - Vetor de com as respostas esperadas (tamanho m) 
    """
    return np.sum((y-predict(theta,X))**2)/len(y)

def gradient_descent(X,y,coef,alpha=0.01,ninter=100):
    
"""
Entrada
X - matriz de medidas
y - vetor de observações
coef - coeficiente inicial

Saída
coef - coeficiente atualizado pelo método gradiente
cost_history - historico da função custo
coef_history - historico dos coeficientes
"""
    coef = np.asarray(coef)
    cost_history = np.zeros(ninter)
    coef_history = np.zeros((ninter,len(coef)))
    
    for i in range(ninter):
        coef = coef- alpha*(X.T.dot((predict(coef,X))-y))*(2/len(y))
        coef_history[i,:] = coef
        cost_history[i] = cost(coef,X,y)
        
    return coef, cost_history, coef_history

def add_vetor_unit(X):
    return np.stack((np.ones(len(X)),X),axis=0).T