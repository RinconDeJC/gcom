// Compile with: 
// g++ problema2.cpp -lpthread -std=c++11 -o testThread -Wall

#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <thread>

#define PI 3.14159265

using namespace std;
struct dPoint{
    double x;
    double y;
};

int get_random_uncovered(const vector<bool>& covered, size_t N){
    int i = rand() % N;
    while (covered[i%N]){++i;}
    return i%N;
}

/// @brief  Cálculo de un recubrimiento por cuadrados de lado side
/// @param points Puntos a recubrir
/// @param side Lado de los cuadrados con que se recubre
/// @param results Array donde poner el resultado
/// @param index Posición del array donde poner el resultado
void cover(
    const vector<dPoint> *points, 
    double side, 
    vector<size_t> *results, 
    size_t index
    ){
    double side2 = side / 2;
    int N = int(points->size());
    vector<bool> covered(N, false);
    int n_covered = 0;
    size_t balls_used = 0;
    while (n_covered < N){
        int index = get_random_uncovered(covered, N);
        ++balls_used;
        covered[index] = true;
        ++n_covered;
        int j = index + 1;
        while (j < N && (*points)[index].x + side2 > (*points)[j].x){
            if (abs((*points)[index].y - (*points)[j].y) < side2 &&
                !covered[j])
            {
                ++n_covered;
                covered[j] = true;
            }
            ++j;
        }
        j = index - 1;
        while (j >= 0 && (*points)[index].x - side2 < (*points)[j].x){
            if (abs((*points)[index].y - (*points)[j].y) < side2 &&
                !covered[j]){
                ++n_covered;
                covered[j] = true;
            }
            --j;
        }
    }
    cout << "Balls_used: " << balls_used << " side: " << side << '\n';
    (*results)[index] = balls_used;
}

/// @brief Cálculo de la dimensión de Hausdorff de un fractal dado como un conjunto 
/// discreto de puntos
/// @param points Fractal representado como un conjunto discreto de puntos
/// @param delta Precisión máxima con la que tomar los recubrimientos.
/// Nótese que si la precisión es muy alta el resultado convergerá a lower bound, 
/// ya que la dimensión de Hasudorff de cualquier conjunto finito de puntos es siempre 0.
/// @param reps Número de veces que tomar un recubrimiento para buscar el ínfimo
/// @param lower_bound Cota inferior de la dimensión de Hausdorff del fractal
/// @param upper_bound Cota superior de la dimensión de Hausdorff del fractal
/// @param max_search Número de veces que hay que iterar la búsqueda binaria
/// @return Aproximnación de la dimensión de Hausdorff del fractal representado por points
double hausdorff_dimension(
    const vector<dPoint> points, 
    double delta,
    int reps,
    double lower_bound,
    double upper_bound,
    int max_search)
{
    // Para paralelizar los cálculos
    vector<size_t> results1(reps);
    vector<size_t> results2(reps);
    vector<std::thread> threads;
    // Primero se calcula el número de bolas para el recubrimiento
    // mínimo con dos diámetros próximos a la precisión pedida
    size_t min_cover1;
    size_t min_cover2;
    double side1 = delta*5; // Precisión algo menor
    double side2 = delta;   // Precisión pedida
    for (int i = 0; i < reps; ++i){
        // Lanzar recubrimientos concurrentes
        threads.push_back(thread(cover, &points, side1, &results1, i));
        threads.push_back(thread(cover, &points, side2, &results2, i));
    }
    // Esperar a todos los threads
    for (auto& th : threads) th.join();  

    // Calculamos las bolas necesarias para el recubrimiento mínimo
    min_cover1 = *std::min_element(results1.begin(), results1.end());
    min_cover2 = *std::min_element(results2.begin(), results2.end());

    // Búsqueda binaria
    // Si es creciente se busca hacia arriba
    // Si es decreciente se busca hacia abajo
    // Se deja un número de iteraciones máxima
    double s;
    for(int i = 0; i < max_search; ++i){
        s = (lower_bound + upper_bound) / 2;
        
        double difference = 
            min_cover2 * pow(side2 * sqrt(2), s) - 
            min_cover1 * pow(side1 * sqrt(2), s);
        if (difference == 0) {
            break;
        }
        if (difference > 0) {
            lower_bound = s;
        }
        else{
            upper_bound = s;
        }
    }
    return s;
}

// --------------------------------------------
// |   Creación del Triángulo de Sierpinski   |
// --------------------------------------------

float sign (dPoint p1, dPoint p2, dPoint p3)
{
    return (p1.x-p3.x)*(p2.y-p3.y) - (p2.x-p3.x)*(p1.y-p3.y);
}

bool PointInTriangle (dPoint pt, dPoint v1, dPoint v2, dPoint v3)
{
    float d1, d2, d3;
    bool has_neg, has_pos;

    d1 = sign(pt, v1, v2);
    d2 = sign(pt, v2, v3);
    d3 = sign(pt, v3, v1);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}

vector<dPoint> Sierpinski_points(
    size_t n_points,
    vector<dPoint> v)
{
    vector<dPoint> triangle;
    float x0 = 1;
    float y0 = 1;
    // Se toma un punto inicial aleatorio dentro del triángulo
    // Realmente para obtener una buena aproximación ni siquiera
    // es necesario que el punto inicial esté en el triángulo, pero 
    // bueno, no nos cuesta nada y es preferible
    while(!PointInTriangle({x0, y0}, v[0], v[1], v[2])){
        x0 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        y0 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    dPoint curr = {x0, y0};
    for (size_t i = 0; i < n_points; ++i){
        // Elegimos un vértice aleatorio y tomamos el siguiente punto
        // como el punto medio entre dicho vértice y el punto actual
        int index = rand() % 3;
        double x_next = (curr.x + v[index].x) / 2;
        double y_next = (curr.y + v[index].y) / 2;
        triangle.push_back({x_next, y_next});
        curr = {x_next, y_next};
    }
    return triangle;
}
bool compare_dPoint(const dPoint& p1, const dPoint& p2){
    if (p1.x == p2.x)
        return p1.y < p2.y;
    return p1.x < p2.x;
}
int main() {
    srand(time(NULL));
    vector<dPoint> vertices;
    // Vértices iniciales del triángulo
    vertices.push_back({0.5, sin(PI/3)});
    vertices.push_back({0,0});
    vertices.push_back({1,0});

    vector<dPoint> triangle = Sierpinski_points(4000000, vertices);
    // Se ordenan según el orden lexicográfico de las coordenadas (x,y)
    sort(triangle.begin(), triangle.end(), compare_dPoint);
    cout << "haursdorff dimension = " << 
        hausdorff_dimension(
            triangle,
            0.0016,
            10,
            1.0,
            2.0,
            10
        ) << '\n';
    return 0;
}