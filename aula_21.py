import numpy as np

# Matriz A em [A]{x}={b}
A = np.array([[3,-0.1,-0.2],[0.1,7,-0.3],[0.3,-0.2,10]])

# vetor {b}
b = np.array([[7.85],[-19.3],[71.4]])

# chute inicial
x = np.zeros((3,1))
xnew = np.zeros((3,1))

tol = 1e-10

# Máximo de iterações
p = 100

for i in range(p):
    
    # Método de Jacobi
    xnew[0] = (b[0] - A[0,1]*x[1] - A[0,2]*x[2]) / A[0,0]
    xnew[1] = (b[1] - A[1,0]*x[0] - A[1,2]*x[2]) / A[1,1]
    xnew[2] = (b[2] - A[2,0]*x[0] - A[2,1]*x[1]) / A[2,2]
    
    # Erro
    err = max(abs((xnew-x)/xnew))
    
    # Atualizar
    x = np.copy(xnew)
    
    if err<=tol:
        break

print(xnew)
print(err)
print(i)
