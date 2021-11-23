from numpy.core.numeric import indices
from numpy.linalg import cond
import numpy as np
import math


# -*- coding: utf-8 -*-


def plota(N, Inc):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Numero de membros
    nm = len(Inc[:, 0])

#    plt.show()
    fig = plt.figure()
    # Passa por todos os membros
    for i in range(nm):

        # encontra no inicial [n1] e final [n2]
        n1 = int(Inc[i, 0])
        n2 = int(Inc[i, 1])

        plt.plot([N[0, n1-1], N[0, n2-1]], [N[1, n1-1],
                                            N[1, n2-1]], color='r', linewidth=3)

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def importa(entradaNome):
    import xlrd



    arquivo = xlrd.open_workbook(entradaNome)

    # Ler os nos
    nos = arquivo.sheet_by_name('Nos')

    # Numero de nos
    nn = int(nos.cell(1, 3).value)

    # Matriz dos nós
    N = np.zeros((2, nn))

    for c in range(nn):
        N[0, c] = nos.cell(c+1, 0).value
        N[1, c] = nos.cell(c+1, 1).value

    # Ler a incidencia
    incid = arquivo.sheet_by_name('Incidencia')

    # Numero de membros
    nm = int(incid.cell(1, 5).value)

    # Matriz de incidencia
    Inc = np.zeros((nm, 4))

    for c in range(nm):
        Inc[c, 0] = int(incid.cell(c+1, 0).value)
        Inc[c, 1] = int(incid.cell(c+1, 1).value)
        Inc[c, 2] = incid.cell(c+1, 2).value
        Inc[c, 3] = incid.cell(c+1, 3).value

    # Ler as cargas
    carg = arquivo.sheet_by_name('Carregamento')

    # Numero de cargas
    nc = int(carg.cell(1, 4).value)

    # Vetor carregamento
    F = np.zeros((nn*2, 1))

    for c in range(nc):
        no = carg.cell(c+1, 0).value
        xouy = carg.cell(c+1, 1).value
        GDL = int(no*2-(2-xouy))
        F[GDL-1, 0] = carg.cell(c+1, 2).value

    # Ler restricoes
    restr = arquivo.sheet_by_name('Restricao')

    # Numero de restricoes
    nr = int(restr.cell(1, 3).value)

    # Vetor com os graus de liberdade restritos
    R = np.zeros((nr, 1))

    for c in range(nr):
        no = restr.cell(c+1, 0).value
        xouy = restr.cell(c+1, 1).value
        GDL = no*2-(2-xouy)
        R[c, 0] = GDL-1

    return nn, N, nm, Inc, nc, F, nr, R


def geraSaida(nome, Ft, Ut, Epsi, Fi, Ti):
    nome = nome + '.txt'
    f = open("saida.txt", "w+")
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


# Matriz de Conectividade -> recebe um excel com os valores de incidência e retorna matrizes normal e transposta.
def matriz_conect(Inc, nn, nm):
    C = np.zeros((nn, nm))
    for i in range(len(Inc)):
        C[int(Inc[i][0]) - 1][i] = -1
        C[int(Inc[i][1]) - 1][i] = 1
    return(C)

# Matriz de Membros ->

def matriz_membros(N, C):
    M = N@C
    return(M)


# Cálculo de comprimento dos elementos -> L

def calculaL(membros):
    col = np.shape(membros)[1]
    L = np.zeros((col, 1))  # Re-ver esse [1]
    for j in range(col):
        L[j] = np.linalg.norm(membros[:, j])
    return L



# Matriz de Rigidez -> recebe um excel com os valores de incidência e retorna matrizes normal e transposta.

def matriz_Ke(matriz_L, C, membros, area, elast, nn):
    col = np.shape(membros)[1]
    lin = np.shape(membros)[0]
    se = np.zeros((2, 2))
    kg = np.zeros((nn*2, nn*2))  

    ke_list = []

    for i in range(col):
        mem_e = (membros[:, i]).reshape(lin, 1)
        mem_e_t = mem_e.T
        se = ((elast*area)/matriz_L[i]) * ((np.matmul(mem_e, mem_e_t))/(np.linalg.norm(mem_e))**2)
        conect_e = C[:, i].reshape(np.shape(C)[0], 1)
        conect_e_t = conect_e.T
        ke = np.kron(np.matmul(conect_e, conect_e_t), se)
        ke_list.append(ke)
        kg += ke

    return kg, ke


def cond_contorno(kg,R):
    if np.shape(kg)[1] == 1:
        return np.delete(kg, list(R[:,0].astype(int)),axis=0)

    return np.delete(np.delete(kg, list(R[:,0].astype(int)),axis=0), list(R[:,0].astype(int)),axis=1)

def jacobi(a, b, tol):
    lin = np.shape(a)[0]
    col = np.shape(a)[1]
    desloc = np.zeros((lin,1))
    desloc_new = np.zeros((lin,1))
    p = 100 

    for i in range(p):
        for l in range(lin):
            for c in range(col):
                if l != c:
                    desloc_new[l] += a[l][c]*desloc[c]

            desloc_new[l] = (b[l] - desloc_new[l])/a[l,l]
        
        err = max(abs((desloc_new-desloc)/desloc_new))
        desloc = np.copy(desloc_new)
        if err<=tol:
            print('Conv ',i)
            break
    return desloc


def calc_ang_elemts(N, membros, L):
    col = np.shape(membros)[1]
    result = np.zeros((col, 4))

    for i in range(col):
        result[i,2] = (membros[0,i])/L[i]
        result[i,0] = -result[i,2]
        result[i,3] = (membros[1,i])/L[i]
        result[i,1] = -result[i,3]
    return result




def desloc_complt(R, desloc):
    u_c = np.zeros((len(R)+len(desloc),1))
    r = list(R[:,0].astype(int))
    var = 0 

    for i in range (len(u_c)):
        if i not in r:
            u_c[i] = desloc[var]
            var += 1
    return u_c

def calc_deformacao(L, Inc, desloc_complt, ele_angs):
    deform_list= np.zeros((len(L), 1))
    
    for i in range(len(L)):
        values = [(int(Inc[i, 0])-1)*2, (int(Inc[i, 0])-1)*2 +1, (int(Inc[i, 1])-1)*2, int(Inc[i, 1]-1)*2 +1]
        d= (1/L[i])*(ele_angs[i]@desloc_complt[values])
        deform_list[i]=d[0]
    return deform_list


def calc_tensao(E,d):
    return int(E)*d

def forca_int(A, tensao):
    return tensao * A

def calc_r_apoio(desloc_complt, kg, R):
    R = list(R[:,0].astype(int))
    return ((kg@desloc_complt)[R])



