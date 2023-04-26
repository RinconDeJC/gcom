#include <fstream>
#include <iostream>
#include <algorithm>
#include <queue>
#include <string>
#include <vector>
#include <unordered_set>
#include <climits>
#include <cmath>

#include <chrono>

#include "UnionFind.h"

using namespace std;
using namespace std::chrono;
// -----------------------------
// Estructuras de datos básicas
// -----------------------------
struct Point{
    int x;
    int y;
    Point(int _x, int _y):
        x(_x),
        y(_y)
        {};
    Point(const Point& p):
        x(p.x),
        y(p.y)
        {};
    string to_string() const {
        return "(" + std::to_string(x) + "," + std::to_string(y)+")";
    }
};

struct Segment{
    Point p1;
    Point p2;
    Segment(int x1=0, int y1=0, int x2=0, int y2=0):
        p1(x1, y1),
        p2(x2,y2)
        {};
    Segment(const Point& _p1, const Point& _p2):
        p1(_p1),
        p2(_p2)
        {};
    string to_string() const {
        return "[" + p1.to_string() + ", " + p2.to_string() + "]";
    }
};
#define compPoint(p1, p2) (p1.x == p2.x && p1.y == p2.y)

class EqualOperator{
public:
    bool operator()(const pair<Segment, int>& s1, const pair<Segment, int> & s2) const{
        return  compPoint(s1.first.p1, s2.first.p1) && compPoint(s1.first.p2, s2.first.p2)
                ||
                compPoint(s1.first.p1, s2.first.p2) && compPoint(s1.first.p2, s2.first.p1);
    }
};
class Hasher {
public:
    size_t operator() (const pair<Segment, int>& s) const {     // the parameter type should be the same as the type of key of unordered_map
        size_t hash = s.first.p1.x;
        hash *= 37;
        hash += s.first.p1.y;
        hash *= 37;
        hash += s.first.p2.x;
        hash *= 37;
        hash += s.first.p2.y;
        return hash;
    }
};

using SegSet = unordered_set<pair<Segment, int>, Hasher, EqualOperator>; 

// -----------------------------
// Manipulacion de datos externos
// -----------------------------
void load_points(const string &file_name, vector<Segment> &segs){
    ifstream in(file_name);
    if (!in.is_open()) throw std::ios_base::failure("No existe el fichero " + file_name);
    int n_segments;
    segs.clear();
    in >> n_segments;
    for (int i = 0; i < n_segments; ++i){
        // Los segmentos tienen que estar escritos en el archivo como
        // x1, y1, x2, y2
        // donde cada uno de esos valores será un entero
        int x1, x2, y1, y2;
        in >> x1;
        in >> y1;
        in >> x2;
        in >> y2;
        segs.push_back(Segment(x1, y1, x2, y2));
    }
}

void save_points(const string &file_name, const vector<Segment> &segs){
    ofstream out(file_name);
    if (!out.is_open()) throw std::ios_base::failure("Error al abrir el fichero " + file_name);
    int n_segments = segs.size();
    out << n_segments << '\n';
    for (int i = 0; i < n_segments; ++i){
        // Los segmentos tienen que estar escritos en el archivo como
        // x1, y1, x2, y2
        // donde cada uno de esos valores será un entero
        int x1, x2, y1, y2;
        out << segs[i].p1.x << " ";
        out << segs[i].p1.y << " ";
        out << segs[i].p2.x << " ";
        out << segs[i].p2.y << "\n";
    }
    out.close();
}

void create_random_segments(
    vector<Segment> &segs, 
    int N, 
    const int range, 
    const int variation
){
    N = N - N % 5;
    int N1 = N / 5;
    vector<int> xrango1(N1);
    for (int i = 0; i < N1; ++i){
        xrango1[i] = rand() % range;
    }
    vector<int> xrango2(N1);
    for (int i = 0; i < N1; ++i){
        xrango2[i] = 10 + rand() % variation;
    }
    vector<int> yrango1(N1);
    for (int i = 0; i < N1; ++i){
        yrango1[i] = rand() % range;
    }
    vector<int> yrango2(N1);
    for (int i = 0; i < N1; ++i){
        yrango2[i] = 10 + rand() % variation;
    }
    vector<int> X;
    vector<int> Y;
    for (int j = 0; j < N1; ++j){
        for (int i = 0; i < 5; ++i){
            X.push_back(xrango1[j] + rand() % xrango2[j]);
            X.push_back(xrango1[j] + rand() % xrango2[j]);
            Y.push_back(yrango1[j] + rand() % yrango2[j]);
            Y.push_back(yrango1[j] + rand() % yrango2[j]);
        }
    }
    segs.clear();
    segs.resize(N);
    for (int i = 0; i < N; ++i){
        segs[i] = Segment(X[2*i], Y[2*i], X[2*i+1], Y[2*i+1]);
    }
    for (auto seg : segs){
        std::cout << seg.to_string() << '\n';
    }
}

// -----------------------------
// Codigo para intersecciones
// -----------------------------

bool aligned(const Point &A, const Point &B, const Point &C){
    int ABx = B.x - A.x;
    int ABy = B.y - A.y;
    int ACx = C.x - A.x;
    int ACy = C.y - A.y;
    if (ABx*ACy-ACx*ABy == 0){
        return C.x <= max(A.x, B.x) && C.x >= min(A.x, B.x);
    }
    else{
        return false;
    }
}

bool coincide(const Segment &seg1, const Segment &seg2){
    return 
        aligned(seg1.p1,seg1.p2,seg2.p1) ||
        aligned(seg1.p1,seg1.p2,seg2.p2) ||
        aligned(seg2.p1,seg2.p2,seg1.p1) ||
        aligned(seg2.p1,seg2.p2,seg1.p2);
}

#define ccw(A, B, C) ((C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x))

bool intersect(const Segment &seg1, const Segment &seg2){
    return  (ccw(seg1.p1,seg2.p1,seg2.p2) != ccw(seg1.p2,seg2.p1,seg2.p2)
            &&
            ccw(seg1.p1,seg1.p2,seg2.p1) != ccw(seg1.p1,seg1.p2,seg2.p2))
            ||
            coincide(seg1, seg2);
}

// -------
// Greddy 
// -------

int joins_n_CC(
    const vector<Segment> &segs, 
    const SegSet &added_segs, 
    const Segment new_seg, 
    UnionFind &CC
){
    int last_segment = CC.nodes();
    CC.add_node();
    int curr_CC = CC.components();
    const int N = segs.size(); 
    for (int i = 0; i < N; ++i){
        if (intersect(segs[i], new_seg)){
            CC.node_union(i, last_segment);
        }
    }
    for (auto seg_pair : added_segs){
        if (intersect(seg_pair.first, new_seg)){
            CC.node_union(seg_pair.second, last_segment);
        }
    }
    return curr_CC - CC.components();
}

void  update_greedy(
    const SegSet &new_segments,
    const vector<Segment> &segs, 
    const Segment seg, 
    const UnionFind &CC,
    int &best_value,
    Segment &best_seg)
{
    if (new_segments.count({seg, 0})) return;
    auto uf = UnionFind(CC);
    int solution = joins_n_CC(segs, new_segments, seg, uf);
    if (solution > best_value){
        best_value = solution;
        best_seg = seg;
    }
}
SegSet greedy(const vector<Segment> &segs, UnionFind CC){
    std::cout << "Starting with " << CC.components() << " componenets\n";
    const auto N = segs.size();
    SegSet new_segments(0);
    while(CC.components() > 1){
        int best_value = 0;
        Segment best_seg(0, 0, 0, 0);
        for (int i = 0; i < N; ++i){
            for (int j = 0; j < i; ++j){
                if (CC.find(i) != CC.find(j)){
                    // Try every pair of points
                    update_greedy(new_segments, segs, Segment(segs[i].p1, segs[j].p1), CC, best_value, best_seg);
                    update_greedy(new_segments, segs, Segment(segs[i].p1, segs[j].p2), CC, best_value, best_seg);
                    update_greedy(new_segments, segs, Segment(segs[i].p2, segs[j].p1), CC, best_value, best_seg);
                    update_greedy(new_segments, segs, Segment(segs[i].p2, segs[j].p2), CC, best_value, best_seg);
                }
            }
            if (i % 100 == 0){
                std::cout << "i = " << i << '\n';
            }
        }
        new_segments.insert({best_seg, CC.nodes()});
        joins_n_CC(segs, new_segments, best_seg, CC);
        std::cout << "Added: " << new_segments.size() << " segments. CC left: " << CC.components() << "\n";
    }
    return new_segments;
}

// -----------------
// Branch and Bound
// -----------------

size_t cota_optimista(size_t added_already, size_t CC_left, size_t last_upgrade){
    if (added_already == 0){
        return INT_MAX - 1;
    }
    else{
        // Para saber cuántos segmentos hacen falta como mínimo hay que 
        // tener en cuenta que si en un momento dado el mejor une Y componentes
        // conexas, a lo sumo el siguiente segmento puede unir Y + 1.
        // Si nos quedan X componenetes conexas esto supone hallar el r tal que:
/*
    \sum_{k=1}^{r-1} (Y+k) < x \leq \sum_{k=1}^{r} (Y+k)
    sii
    r^2-r+2Yr < 2x \leq r^2+r+2Yr
    Y para ello podemos hallar el ceil de la solución con radicando positivo de 
    r = \frac{-2Y-1+\sqrt{4Y^2+4Y+1+8X}}{2}
*/
        float X = float(CC_left);
        float Y = float(last_upgrade);
        size_t solution = std::ceil(0.5*(-2*Y-1+std::sqrt(4*Y*(Y+1)+1+8*X))); 
        return added_already + solution;
    }
}

struct solucion {

    UnionFind CC;
    SegSet added_segs;
    SegSet forbidden_segs;
    int coste_estimado; // cota superior de la mejor solución alcanzable
                            // desde esta solución parcial

    solucion(
        UnionFind comps, 
        SegSet segs,
        SegSet forbidden,
        int bound) 
        :
        added_segs(segs),
        CC(comps),
        forbidden_segs(forbidden),
        coste_estimado(bound)
    {};

    solucion(
        UnionFind comps, 
        SegSet segs,
        Segment newSegment,
        SegSet forbidden,
        size_t bound) 
        :
        added_segs(segs),
        CC(comps),
        forbidden_segs(forbidden),
        coste_estimado(bound)
    {   
        added_segs.insert({newSegment, CC.nodes()-1});
    }

};

class ComparaSoluciones {
public:
  bool operator() (const solucion& s1, const solucion& s2) const
  {
    if (s1.coste_estimado == s2.coste_estimado){
        // Pequeñas heurística para romper empates
        return s1.CC.components() > s2.CC.components();
    }
    return s1.coste_estimado > s2.coste_estimado;
  }
};

struct Candidate{
    UnionFind CC;
    Segment addition;
    int bound;
};

SegSet branch_and_bound(
    const vector<Point>& points, 
    const vector<Segment> &segs, 
    UnionFind firstCC, 
    int bound=-1){
    // Iniciar las soluciones con una única solución del voraz
    SegSet set_solution = greedy(segs, UnionFind(firstCC));
    size_t best_solution = set_solution.size();
    priority_queue<solucion, vector<solucion>, ComparaSoluciones> q;
    solucion start(UnionFind(firstCC), SegSet(0), SegSet(0), 0);
    q.push(start);
    std::cout << "greedy_solution: " << best_solution << '\n';
    std::cout << "start.coste_estimado: " << start.coste_estimado << '\n';
    std::cout << "q.top().coste_estimado: " << q.top().coste_estimado << '\n';
    while (!q.empty() && q.top().coste_estimado < best_solution){
        // Considerar los N^2 segmentos posibles
        // Para cada uno se mira si es prometedor, es decir, 
        // si optimista < mejor_solucion. Para mirar si es prometedor ya se ha 
        // calculado el UF. Después se mete en el array que se ordenará después
        // Una vez se tiene el array ordenado se van considerando si se coge o 
        // no y se mete en la cola añadiendo los nodos no cogidos como segmentos
        // prohibidos. Esto debería quitar muchísimas simetrías.
        solucion actual = q.top();
        std::cout << "q: " << q.size() << "\n";
        std::cout << "cota: " << actual.coste_estimado << "\n";
        std::cout << "mejor: " << best_solution << "\n\n";
        q.pop();
        // Se comprueba si ya es solución
        if (actual.CC.components() == 1){
            set_solution = SegSet(actual.added_segs);
            best_solution = set_solution.size();
            continue;
        }
        vector<size_t> indices;
        vector<Candidate> candidates;
        // Recorrer los segmentos
        for (auto i = 0; i < points.size() - 1; ++i){
            for (auto j = i+1; j < points.size(); ++j){
                Segment newSeg = {points[i], points[j]};
                if (actual.forbidden_segs.count({newSeg, 0}))
                    continue;
                if (actual.CC.find(i / 2) != actual.CC.find(j / 2)){
                    // Mirar si es prometedor
                    UnionFind newCC(actual.CC);
                    // Con esta función vemos cuántas componenetes
                    // conexas se unen y además obtenemos un UnionFind actualizado
                    int last_upgrade = joins_n_CC(
                                            segs, 
                                            actual.added_segs,
                                            {points[i], points[j]},
                                            newCC);

                    int bound = cota_optimista(
                            actual.added_segs.size() + 1,
                            newCC.components(),
                            last_upgrade); 
                    if (bound < best_solution)
                    {   
                        // Si es prometedor lo añadimos a los candidatos
                        indices.push_back(indices.size());
                        candidates.push_back({
                            UnionFind(newCC), 
                            {points[i], points[j]},
                            bound});
                    }
                }
            }
        }
        // Es necesario ordenar para preservar la veracidad de las cotas
        sort(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b){
            return a.bound < b.bound;
        });
        for (auto candidate : candidates){
            q.push({
                UnionFind(candidate.CC),
                SegSet(actual.added_segs),
                candidate.addition,
                SegSet(actual.forbidden_segs),
                candidate.bound
            });
            // Las siguiente soluciones tendrán prohibido emplear
            // los anteriores nodos. De esta forma se preserva 
            // la veracidad de la cota
            actual.forbidden_segs.insert({candidate.addition, 0});
        }
    }
    return set_solution;
}

int test_times(){
    int n = 10;
    size_t sizes[10] = {10, 20, 40, 80, 150, 250, 350, 450, 550, 600};
    vector<size_t> times(10);
    std::fill(times.begin(), times.end(), 0);
    for (int i = 0; i < 10; ++i){
        cout << "\nSIZE: " << sizes[i] << "\n";
        for (int j = 0; j < n; ++j){
            vector<Segment> segs;
            create_random_segments(segs, sizes[i], 1000, 300);
            UnionFind uf = UnionFind(sizes[i]);
            for (int k = 0; k < sizes[i] - 1; ++k){
                for (int l = k+1; l < sizes[i]; ++l){
                    if (intersect(segs[i], segs[j])){
                        uf.node_union(i, j);
                    }
                }
            }
            auto start = high_resolution_clock::now();
            greedy(segs, uf);
            auto stop = high_resolution_clock::now();
            auto duration_ms = duration_cast<milliseconds>(stop - start);
            times[i] += duration_ms.count();
        }
        times[i] /= 10;
    }
    for (int i = 0; i < 10; ++i){
        cout << "Size: " << sizes[i] << " Duration: " << times[i] << '\n';
    }
    ofstream out("times.txt");
    if (!out.is_open()) throw std::ios_base::failure("Error al abrir el fichero times.txt");
    out << 10 << '\n';
    for (int i = 0; i < 10; ++i){
        out << sizes[i] << " " << times[i] << '\n';
    }
    out.close();
    return 0;

}

int main(int argc, char** argv){
    srand(time(NULL));
    vector<Segment> segs;

    // create_random_segments(segs, stoi(argv[1]), 1000, 300);
    // save_points("original_1000.txt", segs);
    load_points("original_data_1000.txt", segs);
    int N = segs.size();
    std::cout << "N=" << N << '\n';
    UnionFind uf = UnionFind(N);
    auto start = high_resolution_clock::now();

    for (int i = 0; i < N; ++i){
        for (int j = i+1; j < N; ++j){
            if (intersect(segs[i], segs[j])){
                uf.node_union(i, j);
            }
        }
    }
    auto stop = high_resolution_clock::now();
    std::cout << "Componentes conexas: " << uf.components() << "\n";
    auto duration_ms = duration_cast<milliseconds>(stop - start);
    
    cout << "duration = " << duration_ms.count() << " ms\n";
    vector<Segment> copy_segs(segs);
    UnionFind copy_uf(uf);
    
    start = high_resolution_clock::now();
    auto solution = greedy(segs, uf);
    stop = high_resolution_clock::now();

    for (auto seg : solution){
        segs.push_back(seg.first);
    }

    save_points("other_original_1000.txt", segs);
    vector<Point> points;
    for (auto seg : copy_segs){
        points.push_back({seg.p1});
        points.push_back({seg.p2});
    }
    auto start = high_resolution_clock::now();
 
    auto b_b_solution = branch_and_bound(points, copy_segs, copy_uf);
    // Get ending timepoint
    auto stop = high_resolution_clock::now();
    for (auto seg: b_b_solution){
        copy_segs.push_back(seg.first);
    }
    N = copy_segs.size();
    UnionFind uf_checker = UnionFind(N);

    for (int i = 0; i < N; ++i){
        for (int j = i+1; j < N; ++j){
            if (intersect(copy_segs[i], copy_segs[j])){
                uf_checker.node_union(i, j);
            }
        }
    }
 
    cout << "Solucion greedy: " << solution.size() << '\n';
    cout << "solucion B&B: " << b_b_solution.size() << '\n';
    save_points("greedy.txt", segs);
    save_points("b_b.txt", copy_segs);
    // Get duration. Substart timepoints to
    // get duration. To cast it to proper unit
    // use duration cast method
    auto duration_min = duration_cast<minutes>(stop - start);
    
    cout << "duration = " << duration_min.count() << " minutes\n";
    cout << "Componenetes conexas tras check: " << uf_checker.components() << '\n';
    return 0;
}