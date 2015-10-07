from matplotlib import pyplot as plt
from floater import hexgrid
import numpy as np
from scipy.spatial import qhull

ha = hexgrid.HexArray(shape=(10,10))
pos = np.array([ha.pos(n) for n in range(ha.N)])
x, y = pos.T
nmid = ha.N/2 + ha.Nx/2
r = -np.sqrt((x - x[nmid])**2 + (y-y[nmid])**2)
mask = r<=3
plt.scatter(*pos.T, c=mask, cmap='Greys')
hr = hexgrid.HexArrayRegion(ha)

bpos = np.array([ha.pos(n) for n in hr.interior_boundary()])
ebpos = np.array([ha.pos(n) for n in hr.exterior_boundary()])
plt.scatter(*bpos.T, c='m')
plt.scatter(*ebpos.T, c='c')


hull = qhull.ConvexHull(bpos)
hv = np.hstack([hull.vertices, hull.vertices[0]])
plt.plot(*hull.points[hv].T, color='k')


def pnpoly(vertx, verty, testx, testy):
    nvert = len(vertx)
    i = 0
    j = nvert -1
    c = False
    while (i < nvert):
    #for (i = 0, j = nvert-1; i < nvert; j = i++) {
        if ( ((verty[i]>testy) != (verty[j]>testy)) and
             (testx < (vertx[j]-vertx[i]) * (testy-verty[i])
                      / (verty[j]-verty[i]) + vertx[i]) ):
            c = not c
        j = i
        i += 1
    return c

xv = hull.points[hull.vertices,0]
yv = hull.points[hull.vertices,1]
test_in_hull = np.zeros_like(r)
for n, (x0, y0) in enumerate(zip(x, y)):
    test_in_hull[n] = pnpoly(xv, yv, x0, y0)
