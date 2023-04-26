import os
import numpy as np
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection


def recursive_plot_Sierpinski(ax, depth, p1, p2, p3):
    if depth == 0:
        return []
    middle1 = np.array([(p1[0]+p2[0])/2,(p1[1]+p2[1])/2])
    middle2 = np.array([(p2[0]+p3[0])/2,(p2[1]+p3[1])/2])
    middle3 = np.array([(p3[0]+p1[0])/2,(p3[1]+p1[1])/2])
    segs = [[middle1, middle2], [middle2, middle3], [middle3, middle1]]
    segs = segs + recursive_plot_Sierpinski(ax, depth-1, p1, middle1, middle3)
    segs = segs + recursive_plot_Sierpinski(ax, depth-1, middle1, p2, middle2)
    segs = segs + recursive_plot_Sierpinski(ax, depth-1, middle3, middle2, p3)
    if len(segs) > 10000:
        print('plotting')
        line_segments = LineCollection(np.array(segs), linestyle='solid', color='black')
        ax.add_collection(line_segments)
        segs = []
    return segs

    
def random_plot_Sierpinski(ax, vertices, n_points):
    x0 = np.random.uniform(vertices[1][0], vertices[2][0])
    y0 = np.random.uniform(vertices[1][1], vertices[0][1])
    curr = np.array([x0, y0])
    points = []
    for _ in range(n_points):
        i = np.random.choice([0,1,2])
        next = np.array([curr[0]+vertices[i][0], curr[1]+vertices[i][1]])/2
        points.append(next)
        curr = next
        if (len(points) > 10000):
            np_points = np.array(points)
            ax.scatter(np_points[:,0], np_points[:,1], s=.01, color='black')
            points = []
    np_points = np.array(points)
    ax.scatter(np_points[:,0], np_points[:,1], s=.01, color='black')

def Sierpinski_points(vertices, n_points):
    x0 = np.random.uniform(vertices[1][0], vertices[2][0])
    y0 = np.random.uniform(vertices[1][1], vertices[0][1])
    curr = np.array([x0, y0])
    points = []
    for _ in range(n_points):
        i = np.random.choice([0,1,2])
        next = np.array([curr[0]+vertices[i][0], curr[1]+vertices[i][1]])/2
        points.append(next)
        curr = next
    return np.array(points)

p1 = np.array([0.5, np.sin(np.pi/3)])
p2 = np.array([0, 0])
p3 = np.array([1,0])
vertices = np.array([p1, p2, p3])

fig, ax = plt.subplots()
a = np.array([[p1, p2], [p2,p3], [p3,p1]])
line_segments = LineCollection(a, linestyle='solid', color='black')
ax.add_collection(line_segments)
segs = recursive_plot_Sierpinski(ax, 10, p1, p2, p3)
line_segments = LineCollection(np.array(segs), linestyle='solid', color='black')
ax.add_collection(line_segments)
ax.set_aspect('equal', adjustable='box')
# plt.show()

fig, ax = plt.subplots()
random_plot_Sierpinski(ax, vertices, 4000000)
ax.set_aspect('equal', adjustable='box')
plt.show()

# Apartado ii)

def get_random_uncovered(covered, N):
    i = np.random.randint(0, N)
    while covered[i%N]:
        i += 1
    return i%N

def cover(points, side):
    side2 = side / 2
    N = points.shape[0]
    covered = [False]*N
    n_covered = 0
    balls_used = 0
    while n_covered < N:
        index = get_random_uncovered(covered, N)
        balls_used += 1
        covered[index] = True
        n_covered += 1
        j = index + 1
        while j < N and points[index][0] + side2 > points[j][0]:
            if  abs(points[index][1] - points[j][1]) < side2 and not covered[j]:
                n_covered += 1
                covered[j] = True
            j += 1
        j = index - 1
        while j >= 0 and points[index][0] - side2 < points[j][0]:
            if  abs(points[index][1] - points[j][1]) < side2 and not covered[j]:
                n_covered += 1
                covered[j] = True
            j -= 1
    return balls_used



def hausdorff_dimension(
        points, 
        epsilon, 
        reps, 
        lower_bound, 
        upper_bound, 
        max_search):
    side1 = epsilon * 5
    side2 = epsilon
    min_cover1 = np.min([cover(points,  side1) for _ in range(reps)])
    min_cover2 = np.min([cover(points,  side2) for _ in range(reps)])

    for _ in range(max_search):
        s = (lower_bound + upper_bound) / 2

        difference = min_cover2 * pow(side2 * np.sqrt(2), s) -\
                     min_cover1 * pow(side1 * np.sqrt(2), s)
        if difference == 0:
            break
        if difference > 0:
            lower_bound = s
        else:
            upper_bound = s
    return s


points = Sierpinski_points(vertices, 100000)
sorted = np.lexsort((points[:,1],points[:,0]))    
triangle = points[sorted]

print(f'Dimensión de Hausdorff = {hausdorff_dimension(triangle, 0.01, 10, 1.0, 2.0, 10)}')