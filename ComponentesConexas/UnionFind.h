//
//  UnionFind.h
//
//  Implementación de estructura para problemas de conectividad dinámica
//
//  Juan Carlos Díaz
//


#ifndef UNION_FIND_H
#define UNION_FIND

#include <vector>
#include <stack>
#include <stdexcept>


class UnionFind {
private:
  int _size; // Tamaño del grafo
  int _n_components;
  std::vector<int> _parent; // Array con los identificadores de las componentes
                            // conexas
  std::vector<int> _tree_size; // Mantiene el tamaño de cada árbol. Para mejora
                               // de coste

public:
  /**
   * Crea la estructura inicial con el grafo sin aristas
   */
  UnionFind(int size) :
    _size(size),
    _n_components(size),
    _parent(size),
    _tree_size(size)
  {
    // El padre de cada nodo es sí mismo
    for (int i = 0; i < size; ++i) _parent[i] = i;

    // El tamaño de cada árbol es 1
    std::fill(_tree_size.begin(), _tree_size.end(), 1);    
  }

  /**
   * Une dos nodos dados
   * @throws domain_error si un nodo no está en el rango correcto
   */
  void node_union(int p, int q) {
    if (p < 0 || q < 0 || p >= _size || q >= _size)
      throw std::domain_error("node out of range");
    int i = find(p);
    int j = find(q);
    // Si ya pertenecen a la misma componente
    // conexa no hace falta hacer nada
    if (i == j) return;

    --_n_components;
    // Weighted quick-union
    // Para mantener el árbol lo más plano posible
    // siempre se añade el árbol de menos peso al de más
    if (_tree_size[i] < _tree_size[j]) {
      _parent[i] = _parent[j];
      _tree_size[j] += _tree_size[i];
    }
    else {
      _parent[j] = _parent[i];
      _tree_size[i] += _tree_size[j];
    }
  }

  void add_node(){
    ++_n_components;
    _tree_size.push_back(1);
    _parent.push_back(_size);
    ++_size;
  }

  /**
   * Devuelve el identificardor de la componente conexa a la que pertenece p
   */
  int find(int p) {
    int root = p;
    std::stack<int> to_fix;
    while (_parent[root] != root) {
      to_fix.push(root);
      root = _parent[root];
    }
    // Path compression
    // Cada nodo que haya sido examinado se conecta directamente a la raíz
    while (!to_fix.empty()) {
      if (_parent[to_fix.top()] != root) {
        _tree_size[_parent[to_fix.top()]] -= _tree_size[to_fix.top()];
        _parent[to_fix.top()] = root;
      }
      to_fix.pop();
    }
    return root;
  }

  /**
   * Devuelve el tamaño de la componente conexa de p
   */
  int size(int p) {
    int i = find(p);
    return _tree_size[i];
  }

  /**
   * Devuelve el número de componenets conexas
   */
  int components() const {
    return _n_components;
  }

  /**
   * Devuelve verdadero si p y q pertenecen a la misma componente conexa
   */
  bool connnected(int p, int q) {
    return find(p) == find(q);
  }

  /**
   * Devuelve la cantidad de nodos del grafo considerado
  */
  int nodes() const{
    return _size;
  }


};
#endif /* UNION_FIND_H */
