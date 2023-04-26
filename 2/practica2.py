import numpy as np

from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

markers = ['o', '^', 's', 'D', 'v' ,'+']
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

# Carga de datos
file1 = "Personas_en_la_facultad_matematicas.txt"
file2 = "Grados_en_la_facultad_matematicas.txt"
X = np.loadtxt(file1, encoding='latin1')
Y = np.loadtxt(file2, encoding='latin1')
labels_true = Y[:,0]

header = open(file1).readline()

# Apartado i)

results = []
s_coefficients = []

for k in range(2,16):
    results.append(KMeans(n_clusters=k, random_state=0, n_init=10).fit(X))
    labels = results[k-2].labels_
    s_coefficients.append(metrics.silhouette_score(X, labels))

fig, ax = plt.subplots()
ax.plot(range(2,16), s_coefficients)
ax.set_xlabel('Number of clusters')
ax.set_ylabel('S coefficient')
plt.show()

print('El mejor valor de k es :', np.argmax(s_coefficients) + 2)

from scipy.spatial import Voronoi, voronoi_plot_2d

# Obtenemos los centros para k = 3
k = 3
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
fig, ax = plt.subplots()
centers = kmeans.cluster_centers_
voronoi_gen = Voronoi(centers)
voronoi_plot_2d(voronoi_gen, ax=ax)

# Now plot the rest of the points
unique_labels = set(kmeans.labels_)
colors = [plt.cm.Spectral(each)\
          for each in np.linspace(0, 1, len(unique_labels))]

ax.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
problem = np.array([[0, 0], [0, -1]])

plt.plot(problem[:,0],problem[:,1],'o', markersize=12, markerfacecolor="red")
plt.autoscale()
plt.show()

# Apartado ii)
from sklearn.cluster import DBSCAN

n0 = 10
N = 100
random_epsilons = np.sort(np.random.uniform(0.1, 0.4, (N, )))
euclidean_s = []
euclidean_k = []
for epsilon in random_epsilons:
  db = DBSCAN(eps=epsilon, min_samples=n0, metric='euclidean').fit(X)
  labels = db.labels_
  n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
  euclidean_s.append(metrics.silhouette_score(X, labels))
  euclidean_k.append(n_clusters_)

manhattan_s = []
manhattan_k = []
for epsilon in random_epsilons:
  db = DBSCAN(eps=epsilon, min_samples=n0, metric='manhattan').fit(X)
  labels = db.labels_
  n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  
  manhattan_s.append(metrics.silhouette_score(X, labels))
  manhattan_k.append(n_clusters_)



fig, ax = plt.subplots()
ax.plot(random_epsilons, euclidean_k, label='euclidean', color=lighten_color(colors[0]))
ax.plot(random_epsilons , euclidean_k, markers[0], markersize = 4, markerfacecolor = \
        lighten_color(colors[0]), markeredgecolor = colors[0])

ax.plot(random_epsilons, manhattan_k, label='manhattan', color=lighten_color(colors[2]))
ax.plot(random_epsilons , manhattan_k, markers[1], markersize = 4, markerfacecolor = \
        lighten_color(colors[2]), markeredgecolor = colors[2])
ax.set_ylabel('Number of clusters')
ax.set_xlabel('epsilon')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(random_epsilons, euclidean_s, label='euclidean', color=lighten_color(colors[0]))
ax.plot(random_epsilons, euclidean_s, markers[0], markersize = 4, markerfacecolor = \
        lighten_color(colors[0]), markeredgecolor = colors[0])

ax.plot(random_epsilons, manhattan_s, label='manhattan', color=lighten_color(colors[2]))
ax.plot(random_epsilons, manhattan_s, markers[1], markersize = 4, markerfacecolor = \
        lighten_color(colors[2]), markeredgecolor = colors[2])
ax.set_ylabel('S coefficient')
ax.set_xlabel('epsilon')
ax.legend()
plt.show()
best_eps_manhattan = random_epsilons[np.argmax(manhattan_s)]
best_eps_euclidean = random_epsilons[np.argmax(euclidean_s)]
print('Mejor epsilon para Manhattan: ', best_eps_manhattan)
print('Mejor epsilon para Euclidean: ', best_eps_euclidean)

fig, ax = plt.subplots()
ax.scatter(euclidean_k, euclidean_s, label='euclidean', color=lighten_color(colors[0]))

ax.scatter(manhattan_k, manhattan_s, label='manhattan', color=lighten_color(colors[2]))
ax.set_ylabel('S coefficient')
ax.set_xlabel('Number of clusters')
ax.legend()
plt.show()

# Apartado iii)
problem = np.array([[0, 0], [0, -1]])
clases_pred = kmeans.predict(problem)
print(clases_pred)