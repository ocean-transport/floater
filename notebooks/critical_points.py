import numpy as np
import scipy.ndimage as nd
from scipy.spatial import qhull
import matplotlib.path as mplPath

def classify_critical_points(b):
    """Classifying critical points of b.
    Assumes b is on a rectangular grid."""
    res_hex = classify_critical_points_hexgrid(rect_to_hex(b))
    # need to shift the x indices back
    res = {}
    for k, v in res_hex.iteritems():
        y, x = v
        res[k] = hex_to_rect_coords(y, x)
    return res

def classify_critical_points_hexgrid(b):
    """Classifying critical points of b.
    Assumes b is hex-tesselated with the 0th x row offset by +1/2.
    """
    type_dict = {'max': 1, 'min': -1, 'saddle': 2}
    # ignore the others

    cp_l = nd.filters.generic_filter(b, filter_critical_points, size=(3,3))
    cp_r = nd.filters.generic_filter(b, filter_critical_points, size=(3,3),
                                      extra_keywords={'left': False})
    cp = cp_l.copy()
    cp[::2,:] = cp_r[::2,:]

    res = dict.fromkeys(type_dict)
    for k, v in type_dict.iteritems():
        idx = np.nonzero(cp==v)
        res[k] = idx

    return res

def filter_critical_points(b, debug=False, left=True):
    """Filter function for classifying critical points.
    returns sum of hexagonal lattice differences:
     0: regular point
    +1: maximum
    -1: minimum
    +2: saddle point
    -2: zero gradient detected
    -3: monkey point
    """
    
    #assert b.shape == (3,3)
    b.shape = (3,3)
    
    # this is the key
    # but the data apparently has to be pre-processed
    
    # left connectivity
    if left:
        mask = np.array([[0,0,1],[0,1,0],[0,0,1]])
        idx = np.r_[0,1,5,7,6,3]
    else:
        mask = np.array([[1,0,0],[0,1,0],[1,0,0]])
        idx = np.r_[1,2,5,8,7,3]
    
    # this didn't work so good
    #s = np.ma.masked_array(np.sign(b - b[1,1]), mask).compressed()
    
    # manually order the elements
    s = np.sign(b - b[1,1]).ravel()[idx]
    
    ds = np.abs(s - np.roll(s,1))/2
    # the number of sign changes determines the topology of the neighborhood
    #print ds
    dss = ds.sum()

    if debug and (dss!=2) and not np.any(s==0):
        print 'dss: ', dss
        print np.ma.masked_array(b - b[1,1], mask)
        print s
    
    # it is very unlikely that a real image will have an exactly zero difference
    if np.any(s==0):
        return -2
        #raise ValueError('Detected zero gradient')
          
    if dss==0:
        # extremum
        return -s[0]
    elif dss==2:
        # regular point
        return 0
    elif dss==4:
        # saddle
        return 2
    else:
        return -3
        #raise ValueError('Number of sign changes (%g) unexpected' % ds)

def rect_to_hex(f):
    """Transform f (on a rectangular grid) to hexagonal grid."""
    fhex = np.nan * np.ones_like(f)
    # create a shifted array 1/2 pt to the right (x-axis -> axis=-1)
    fshift = 0.5 * (f[...,1:] + f[...,:-1])
    # fill only the even rows (y-axis) with the shifted values
    fhex[...,::2,:-1] = fshift[...,::2,:]
    # odd rows get the original array
    fhex[...,1::2,:] = f[...,1::2,:]
    return fhex


def hex_to_rect_coords(y, x):
    """Take coords x and y (assumed to represent hex grid coords)
    and transform them back to regular coords."""
    
    idx = np.mod(y,2)==0
    xnew = x.copy().astype('f4')
    xnew[idx] -= 0.5
    return xnew, y


### hex grid indexing ###

geom = """
This indexing is called "even-R horizontal offset"
http://www.redblobgames.com/grids/hexagons/


|       |       |       |       |       |       |   
|  3,0  |  3,1  |  3,2  |  3,3  |  ...  |  3,Nx |
 \     / \     / \     / \     / \     / \     / \     
  \   /   \   /   \   /   \   /   \   /   \   /   \ 
   \ /     \ /     \ /     \ /     \ /     \ /     \
    |       |       |       |       |       |       |   
    |  2,0  |  2,1  |  2,2  |  ...  |  ...  |  2,Nx |
   / \     / \     / \     / \     / \     / \     /
  /   \   /   \   /   \   /   \   /   \   /   \   /
 /     \ /     \ /     \ /     \ /     \ /     \ /
|       |       |       |       |       |       |   
|  1,0  |  1,1  |  1,2  |  1,3  |  ...  |  1,Nx |
 \     / \     / \     / \     / \     / \     / \     
  \   /   \   /   \   /   \   /   \   /   \   /   \ 
   \ /     \ /     \ /     \ /     \ /     \ /     \
    |       |       |       |       |       |       |   
    |  0,0  |  0,1  |  0,2  |  ...  |  ...  |  0,Nx |
=====================================================    
    
Consider the cell (1,2).
Its neighbors are (0,1), (1,1), (2,1), (2,2), (1,3), (0,2)
Its cartesian neighborhood is
[(2,1), (2,2), (2,3)]
[(1,1), (1,2), (1,3)]
[(0,1), (0,2), (0,3)]
which contains the non-neighbor points (2,3) and (0,3).
The neighborhood mask should therefore be
[(2,1), (2,2),   X  ]
[(1,1), (1,2), (1,3)]
[(0,1), (0,2),   X  ]
We will call this a LEFT CONNECTED cell.

Now consider the cell (2,1).
Its neighbors are (1,1), (2,0), (3,1), (3,2), (2,2), (1,2)
Its cartesian neighborhood is
[(3,0), (3,1), (3,2)]
[(2,0), (2,1), (2,2)]
[(1,0), (1,1), (1,2)]
which contains the non-neighbor points (1,0) and (3,0).
The neighborhood mask should therefore be
[  X  , (3,1), (3,2)]
[(2,0), (2,1), (2,2)]
[  X  , (1,1), (1,2)]
We will call this a RIGHT CONNECTED cell.

"""

def hexcoords2points(J,I):
    """Assuming an even-r offset horizontal layout.
    Take index coordinates J,I and return spatial coordinates J, Iout"""
    shiftmask = np.mod(J,2)==0
    #Iout = np.asfarray(I)
    Iout = np.asfarray(I)-0.25
    Iout[shiftmask] += 0.5
    return J, Iout

class HexArray(object):
    """Stores a hex-tesselated numpy array."""
    
    def __init__(self, data, already_hex=True):
        
        if already_hex:
            self.data = data
        else:
            # shift the grid
            pass
        
        assert self.data.ndim == 2 
        self.Nx = self.data.shape[-1]
        self.Ny = self.data.shape[0]
        
    def get_neighbors(self, jin, iin):
        """Return the indices and values of neighboring points.
        j and i are arrays of indices."""
                
        # make sure the requested points are in bounds
        assert (np.all(jin>=0) and np.all(iin>=0) and
                np.all(jin<self.Ny) and np.all(iin<self.Nx))
                
        neighbors = set()
        points = set()
        for j, i in zip(
                np.atleast_1d(np.asarray(jin)), 
                np.atleast_1d(np.asarray(iin))):
            points.add((j,i))
            
            # Strategy for boundaries is to limit indices
            # and then remove duplicate points after the neighbor
            # list has been built.
            # The alternative is lots of tedious logic.
            iup = min(i+1, self.Nx-1)
            jup = min(j+1, self.Ny-1)
            idn = max(i-1, 0)
            jdn = max(j-1, 0)

            # figure out if we are in an even or odd row
            even = (j%2 == 0)

            if even:
                neighbors.update(set([
                          (jup,i), (jup,iup),                
                  (j,idn),(j,i),   (j,  iup),
                          (jdn,i), (jdn,iup)
                ]))
            else:
                neighbors.update([
                  (jup,idn),(jup,i),                
                  (j,  idn),(j,i)  ,  (j,  iup),
                  (jdn,idn),(jdn,i)
                ])
                
        # go back through and remove the points themselves
        neighbors.difference_update(points)

        jout, iout = np.array(list(neighbors)).T
        dout = self.data[jout,iout]
        return jout, iout, dout

    def convex_region(self, jin, iin, psign=1, radius=0.0):
        """Find the largest convex curve enclosing a local extremum.
        """
        
        dmax = np.max(self.data[jin,iin])

        # don't want to do it like this because it cloud use a lot of memory
        #regmask = np.zeros(self.shape, dtype=bool)
        #regmask[j,i] = 1
        
        region = [(jin,iin)]
        # This is an awkward data structure.
        # I don't know how many points I will have in the region,
        # so I need an expandable data structure (i.e. list).
        # But most of the numpy function want arrays of I and J indices,
        # not lists of points. So I have to keep switching back and forth.

        is_convex = True
        while is_convex:
            
            #jmask, imask = regmask.nonzero()
            j, i = np.atleast_2d(region).T
            jb, ib, d = self.get_neighbors(j,i)
            # the difference between the extremum and the surrounding points
            # (masking negative values excludes other extrema)
            delta = np.ma.masked_less(psign*(dmax - d),0)
            nxt = np.argmin(delta)

            # need at least five points for a convex shape 
            if len(j)>4:
                # convex hull stuff                
                jpt, ipt = hexcoords2points(j, i)
                jptb, iptb = hexcoords2points(jb, ib)
                points = np.vstack([ipt, jpt]).T
                bpoints = np.vstack([iptb, jptb]).T

                is_convex = test_convex(points, bpoints, radius)
                
            if is_convex:
                region.append((jb[nxt], ib[nxt]))
            else:
                region.pop()
            #print 'len(region)', len(region)
                        
        # finally, make sure that the extremum is not on the boundary of the region
        # this should eliminate most of the filamentary structures
        for jb, ib, d in zip(*self.get_neighbors(jin,iin)):
            if not (jb, ib) in region:
                region = [np.nan, np.nan]
                break
        
        return np.atleast_2d(region).T



def test_convex(points, bpoints, radius=0.0):
    is_convex = True
    # compute the convex hull of the region
    try:
        hull = qhull.ConvexHull(points)
        qhull_success = True
    except:
        # qhull can fail if the points are all on the same line
        # if we have five points on the same line, 
        # these are note the drones we're looking for
        qhull_success = False
        is_convex = False

    # turn into a matplotib polygon
    if qhull_success:
        vertices = [[points[v,0], points[v,1]] for v in hull.vertices]
        # add dummy endpoint (ignored but used to close polygon)
        vertices.append(vertices[-1])
        codes = np.ones(len(vertices)) * mplPath.Path.LINETO
        codes[0] = mplPath.Path.MOVETO
        codes[-1] = mplPath.Path.CLOSEPOLY
        bbPath = mplPath.Path(vertices, codes)

        for bpt in bpoints:
            # the hull should not contain any of the boundary points
            if bbPath.contains_point(bpt, radius=radius):
                is_convex = False
                # need to get rid of the most recently added point,
                # which is the one responsible for breaking convexity
    #print 'test_convex:', is_convex
    return is_convex




