from UnionFind import *
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import time
#Generamos 1000 segmentos aleatorios, pero siempre serán los mismos

#Usaremos primero el concepto de coordenadas
X = []
Y = []
def load_txt(fileName):
    f = open(fileName)
    N = int(f.readline())
    segs = np.zeros((N, 2, 2))
    for i in range(N):
        coords = [int(a) for a in f.readline().split(' ')]
        
        segs[i] = np.array([[coords[0], coords[1]],[coords[2], coords[3]]])
    return segs
#Fijamos el modo aleatorio con una versión prefijada. NO MODIFICAR!!
random.seed(a=1, version=2)

#Generamos subconjuntos cuadrados del plano R2 para determinar los rangos de X e Y
xrango1 = random.sample(range(100, 1000), 200)
xrango2 = list(np.add(xrango1, random.sample(range(10, 230), 200)))
yrango1 = random.sample(range(100, 950), 200)
yrango2 = list(np.add(yrango1, random.sample(range(10, 275), 200)))

for j in range(len(xrango1)):
    for i in range(5):
        random.seed(a=i, version=2)
        xrandomlist = random.sample(range(xrango1[j], xrango2[j]), 4)
        yrandomlist = random.sample(range(yrango1[j], yrango2[j]), 4)
        X.append(xrandomlist[0:2])
        Y.append(yrandomlist[2:4])


# Fin de la plantilla        
segs = np.zeros((len(X), 2, 2), dtype=np.int32)
for i in range(len(X)):
    segs[i] = np.array([[X[i][0], Y[i][0]],[X[i][1], Y[i][1]]])
def save_points(fileName, segs):
    f = open(fileName, 'w')
    f.write(f'{len(segs)}\n')
    for seg in segs:
        f.write(f'{seg[0,0]} {seg[0,1]} {seg[1,0]} {seg[1,1]}\n')
    f.close()
save_points('original_data_1000.txt', segs)
# Comprueba si A, B, y C están en una recta en el orden A,C,B
def aligned(A,B,C):
    ABx = B[0] - A[0]
    ABy = B[1] - A[1]
    ACx = C[0] - A[0]
    ACy = C[1] - A[1]
    if ABx*ACy-ACx*ABy == 0:
        return C[0] <= max(A[0], B[0]) and C[0] >= min(A[0], B[0])
    else:
        return False
    
def coincide(seg1, seg2):
    return  aligned(seg1[0], seg1[1], seg2[0]) or \
            aligned(seg1[0], seg1[1], seg2[1]) or \
            aligned(seg2[0], seg2[1], seg1[0]) or \
            aligned(seg2[0], seg2[1], seg1[1])

# Devuelve si los puntos A, B, C están en orden antihorario
# Es decir, si la pendiente de AB es menor que la de AC
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Devuelve True sii seg1 interseca a seg2
# AB interseca a CD sii A y B están separados por el segmento
# CD y C y D están separados por el segmento AB
# Y si esto último pasa entonces él orden de ACD es distinto
# del de BCD y los mismo para el otro segmento
def intersect(seg1,seg2):
    return  ccw(seg1[0],seg2[0],seg2[1]) != ccw(seg1[1],seg2[0],seg2[1])\
            and\
            ccw(seg1[0],seg1[1],seg2[0]) != ccw(seg1[0],seg1[1],seg2[1])\
            or\
            coincide(seg1, seg2)

def find_CC(segs):

    N = segs.shape[0]
    uf = UnionFind(N)
    inicio = time.time()
    for i in range(N-1):
        for j in range(i+1, N):
            if intersect(segs[i], segs[j]):
                uf.node_union(i, j)
    fin = time.time()
    print(f'tiempo = {fin - inicio}')
    print(f'Hay {uf.components()} componentes conexas')
    return uf

def print_CC(segs, uf):
    N = segs.shape[0]
    colors_dictionary = {}
    value = 0
    colors = [-1] * N
    for i in range(N):
        component = uf.find(i)
        if component in colors_dictionary:
            colors[i] = colors_dictionary[component]
        else:
            colors_dictionary[component] = value
            colors[i] = value
            value += 1
    indexing = np.linspace(0, 1, len(colors_dictionary))
    colors = [indexing[colors[i]] for i in range(N)]
    colors = cm.prism(colors)

    fig, ax = plt.subplots()
    plt.tick_params(left=False, bottom=False)
    fig.set_dpi(250)
    for i in range(N):
        ax.plot(segs[i,:,0], segs[i,:,1], c=colors[i], linewidth=.5)
    plt.show()

def print_added(segs, initial_N):
    N = segs.shape[0]
    print(f'N={N}')
    print(f'initial_N={initial_N}')
    fig, ax = plt.subplots()
    plt.tick_params(left=False, bottom=False)
    fig.set_dpi(250)
    for i in range(initial_N):
        ax.plot(segs[i,:,0], segs[i,:,1], c='r', linewidth=.5, alpha=0.75)
    for i in range(initial_N, N):
        ax.plot(segs[i,:,0], segs[i,:,1], c='b', linewidth=.6)
    plt.show()

a = time.time()
segs_initial = load_txt('original_data_1000.txt')
initial_N = segs_initial.shape[0]
uf = find_CC(segs_initial)
print_CC(segs_initial, uf)
segs = load_txt('greedy_original_1000.txt')
uf = find_CC(segs)
print_added(segs, initial_N)






