import os
import numpy as np
import heapq as hq

def build_tree(heap):
    # Modelo del arbol:
    # 4-uplas donde cada posicione es:
    #   0: (número de apariciones, orden de aparicion)
    #   1: nombre del nodo
    #   2: hijo izquierdo
    #   3: hijo derecho
    while len(heap) > 1:
        item1 = hq.heappop(heap)
        item2 = hq.heappop(heap)
        result = (\
            (item1[0][0] + item2[0][0], min(item1[0][1], item2[0][1])),\
            item1[1] + item2[1],\
            item1,\
            item2)
        hq.heappush(heap, result)
    # El último elemento en la cola de prioridad tendrá el
    # árbol completo construido
    return hq.heappop(heap)

def build_dictionary(tree):
    if tree[2] == None: # Entonces tree[3] también será None
        return {tree[1]: ''}
    left = build_dictionary(tree[2])
    for k in left.keys():
        left[k] = '0' + left[k]
    right = build_dictionary(tree[3])
    for k in right.keys():
        right[k] = '1' + right[k]
    return {**left, **right}

def code_from_text(text, dict):
    code = ''
    for c in text:
        if not c in dict:
            code += '('+c+' no esta en el diccionario)'
        else:
            code += dict[c]
    return code

def text_from_code(code, tree):
    i = 0
    text = ''
    while i < len(code):
        curr_node = tree
        not_found = True
        while not_found:
            try:
                if code[i] == '0':
                    curr_node = curr_node[2]
                else:
                    curr_node = curr_node[3]
                i += 1
                if curr_node[2] == None:
                    not_found = False
            except IndexError:
                # Si el código no es consistente con el árbol dado
                # saltará un índice de error. Devolveremos la palabra
                # decodificada hasta el momento
                return text
        text += curr_node[1]
    return text


os.getcwd()
#os.chdir(ruta)
#files = os.listdir(ruta)

with open('GCOM2023_pract1_auxiliar_eng.txt', 'r',encoding="utf8") as file:
      en = file.read()
     
with open('GCOM2023_pract1_auxiliar_esp.txt', 'r',encoding="utf8") as file:
      es = file.read()


from collections import Counter
tab_en = Counter(en)
tab_es = Counter(es)
# Añadimos el orden de aparicion
for k in tab_en.keys():
    tab_en[k] = [tab_en[k], -1]
for k in tab_es.keys():
    tab_es[k] = [tab_es[k], -1]
for i in range(len(en)):
    if tab_en[en[i]][1] == -1:
        tab_en[en[i]][1] = i
        tab_en[en[i]] = tuple(tab_en[en[i]])
for i in range(len(es)):
    if tab_es[es[i]][1] == -1:
        tab_es[es[i]][1] = i
        tab_es[es[i]] = tuple(tab_es[es[i]])
# Apartado 1
print('Apartado 1')
# Hallar el código de Huffman binario y las logitudes medias de ambos textos.
# Comprobar que se satisface el Teorema de Shannon.

# Primero construimos los árboles binarios
list_en = [(v, k, None, None) for k, v, in tab_en.items()]
list_es = [(v, k, None, None) for k, v, in tab_es.items()]

hq.heapify(list_en)
hq.heapify(list_es)

tree_en = build_tree(list_en)
tree_es = build_tree(list_es)

dict_en = build_dictionary(tree_en)
dict_es = build_dictionary(tree_es)

code_en = code_from_text(en, dict_en)
code_es = code_from_text(es, dict_es)


# Calculamos las longitudes medias:
W_en = np.sum([v[0] for v in tab_en.values()])
W_es = np.sum([v[0] for v in tab_es.values()])

L_en = 1/float(W_en) * (np.sum([v[0] * len(dict_en[k]) for k, v, in tab_en.items()]))
L_es = 1/float(W_es) * (np.sum([v[0] * len(dict_es[k]) for k, v, in tab_es.items()]))
print('Calculo de las longitudes medias:')
print('L(Seng) = ' + str(L_en))
print('L(Sesp) = ' + str(L_es))
print('')

# Calculo de la entropia

prob_en = {k: float(v[0]) / W_en for k, v, in tab_en.items()}
prob_es = {k: float(v[0]) / W_es for k, v, in tab_es.items()}

H_en = -np.sum([v * np.log2(v) for v in prob_en.values()])
H_es = -np.sum([v * np.log2(v) for v in prob_es.values()])

print('Cálculo de la entropia')
print('H(Seng) = ' + str(H_en))
print('H(Sesp) = ' + str(H_es))
print('')

if L_en >= H_en and L_en < H_en + 1 and L_es >= H_es and L_es < H_es + 1:
    print('Se cumple el Teorema de Shannon')
else:
    print('NO se cumple el Teorema de Shannon')
print('')
# Apartado 2
# Codificar la palabra 'dimension' en ambas lenguas

word = 'dimension'
code_en = code_from_text(word, dict_en)
code_es = code_from_text(word, dict_es)
print('Apartado 2')
print('Codificación en inglés : ' + code_en)
print('Codificación en español: ' + code_es + '\n')
long_en = int(len(word) * np.ceil(np.log2(len(tab_en))))
long_es = int(len(word) * np.ceil(np.log2(len(tab_es))))
print('Longitud del código en inglés: ' + str(len(code_en)))
print('Longitud de la codificación usual: ' + str(long_en))
print('Mejora: ' + f'{((long_en / len(code_en) - 1)*100):.2f}' + '%\n')
print('Longitud del código en español: ' + str(len(code_es)))
print('Longitud de la codificación usual: ' + str(long_es))
print('Mejora: ' + f'{((long_es / len(code_es) - 1)*100):.2f}' + '%\n')
# Apartado 3
print('Apartado 3')
code_3 = '0101010001100111001101111000101111110101010001110'
print('Resultado de la decodificación: ' + text_from_code(code_3, tree_en))