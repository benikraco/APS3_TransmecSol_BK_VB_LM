import numpy as np
import xlrd
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# -*- coding: utf-8 -*-
def plota(N,Inc):
    # Numero de membros
    nm = len(Inc[:,0])
    
#    plt.show()
    fig = plt.figure()
    # Passa por todos os membros
    for i in range(nm):
        
        # encontra no inicial [n1] e final [n2] 
        n1 = int(Inc[i,0])
        n2 = int(Inc[i,1])        

        plt.plot([N[0,n1-1],N[0,n2-1]],[N[1,n1-1],N[1,n2-1]],color='r',linewidth=3)


    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
def importa(entradaNome):
    
    arquivo = xlrd.open_workbook(entradaNome)
    
    ################################################## Ler os nos
    nos = arquivo.sheet_by_name('Nos')
    
    # Numero de nos
    nn = int(nos.cell(1,3).value)
                 
    # Matriz dos nós
    N = np.zeros((2,nn))
    
    for c in range(nn):
        N[0,c] = nos.cell(c+1,0).value
        N[1,c] = nos.cell(c+1,1).value
    
    ################################################## Ler a incidencia
    incid = arquivo.sheet_by_name('Incidencia')
    
    # Numero de membros
    nm = int(incid.cell(1,5).value)
                 
    # Matriz de incidencia
    Inc = np.zeros((nm,4))
    
    for c in range(nm):
        Inc[c,0] = int(incid.cell(c+1,0).value)
        Inc[c,1] = int(incid.cell(c+1,1).value)
        Inc[c,2] = incid.cell(c+1,2).value
        Inc[c,3] = incid.cell(c+1,3).value
    
    ################################################## Ler as cargas
    carg = arquivo.sheet_by_name('Carregamento')
    
    # Numero de cargas
    nc = int(carg.cell(1,4).value)
                 
    # Vetor carregamento
    F = np.zeros((nn*2,1))
    
    for c in range(nc):
        no = carg.cell(c+1,0).value
        xouy = carg.cell(c+1,1).value
        GDL = int(no*2-(2-xouy)) 
        F[GDL-1,0] = carg.cell(c+1,2).value
         
    ################################################## Ler restricoes
    restr = arquivo.sheet_by_name('Restricao')
    
    # Numero de restricoes
    nr = int(restr.cell(1,3).value)
                 
    # Vetor com os graus de liberdade restritos
    R = np.zeros((nr,1))
    
    for c in range(nr):
        no = restr.cell(c+1,0).value
        xouy = restr.cell(c+1,1).value
        GDL = no*2-(2-xouy) 
        R[c,0] = GDL-1


    return nn,N,nm,Inc,nc,F,nr,R

def geraSaida(nome,Ft,Ut,Epsi,Fi,Ti):
    nome = nome + '.txt'
    f = open("saida.txt","w+")
    f.write('Reacoes de apoio [N]\n')
    f.write(str(Ft))
    f.write('\n\nDeslocamentos [m]\n')
    f.write(str(Ut))
    f.write('\n\nDeformacoes []\n')
    f.write(str(Epsi))
    f.write('\n\nForcas internas [N]\n')
    f.write(str(Fi))
    f.write('\n\nTensoes internas [Pa]\n')
    f.write(str(Ti))
    f.close()


# nn -> nós
# N  -> matriz de nos
# nm -> num de membros
# Inc -> Matriz de incidencia [saída, chegada, área transversal[m2], módulo de elasticidade[Pa]]
# nc -> num de cargas
# F -> vetor carregamento
# nr -> Numero de restricoes
# R ->  Vetor com os graus de liberdade restritos

[nn,N,nm,Inc,nc,F,nr,R] = importa("entrada.xlsx")

# matriz de conectividade
C = np.zeros((len(Inc), len(N[0])))
for i in range(0, len(Inc)):
    C[i][int(Inc[i][0]) - 1] = -1
    C[i][int(Inc[i][1]) - 1] = 1

# não funciona com C.transpose
# c_transp = c.transpose()
# o correto é C.T

# matriz de conectividade transposta
C_transp = C.T

# print(C)
# print(C_transp)

# matriz de membros e transposta
M = np.matmul(N,C)
M_transp = M.T

# aqui tá certo o tipo e o shape
# print(type(M))
# print(M.shape)

#print(M)
#print(M_transp)

# membro e membro transposto do elemento 1
# ao fazer a seleção dá erro, pois não considera mais como matriz (ver linha 153)
M_e1 = M[:,0]
M_transp_e1 = M_transp[0,:]

# ao printar o shape pode-se ver que a matriz está com dimensão errada
#print(type(M_e1))
#print(M_e1.shape)

#print(M_e1)
#print(M_transp_e1)

# Calculando matriz S

# área transversal e1
A_e1 = Inc[0][2]
# módulo de elasticidade e1
E_e1 = Inc[0][3]
# comprimento do e1
L_e1 = np.linalg.norm(M_e1)

# Rigidez
k = (A_e1 * E_e1) / L_e1

mat_s = k * np.matmul(M_e1,M_transp_e1)/abs(M_e1)**2

#print(mat_s)

# Calculando matriz rigidez e1 
C_e1 = C[:,0]
C_transp_e1 = C_transp[:,0]

#print(C_e1)
#print(C_transp_e1)

mat_k = np.kron(np.matmul(C_e1, C_transp_e1), mat_s)

#print(mat_k)















#-------------------------------------------------------------
# def calculaL(x1, x2, y1, y2):
#     L = np.sqrt(((x2-x1)**2) + ((y2-y1)**2))
#     return L

# L = np.linalg.norm(membros)
# print(L)

# matriz_L = np.zeros()
# for i in range(0, len(matriz_L)):
    


# # norm, modulo de membros = L 
# membros[:,0] # pegar coluna

# matriz_se = np.zeros() #corrigir os valores de criação
# for i in range(): #corrigir os valores de range
#     for j in range(): #corrigir os valores de range
#         matriz_se[i][j]

# def matriz_se(e, a, l, m, e):
#     return ((e*a)/l) * ((m_e*m_e.transpose())/(abs(m_e)**2))
#-------------------------------------------------------------

