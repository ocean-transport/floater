#cython: profile=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#from __future__ import division
import numpy as np
import cython
cimport numpy as np
from libc.stdlib cimport malloc, free
from libcpp.unordered_set cimport unordered_set
from cython.parallel cimport prange, threadid


# integer indices
DTYPE_int = np.int32
ctypedef np.int32_t DTYPE_int_t
# array values
DTYPE_flt = np.float64
ctypedef np.float64_t DTYPE_flt_t

# what type should things be?
# http://stackoverflow.com/questions/18462785/what-is-the-recommended-way-of-allocating-memory-for-a-typed-memory-view
# answer: malloc

# special value for border points
cdef DTYPE_int_t INT_NOT_FOUND = -9999

cdef class HexArray:
    # array shape
    cdef readonly DTYPE_int_t Nx, Ny, N
    cdef readonly bint index_right
    # array data

    def __init__(self, nx, ny, idx_right=1):
        """Initialize new hex array of a given shape."""

        self.Nx = nx
        self.Ny = ny
        self.N = nx*ny
        self.index_right = idx_right

    # this function sucks because it uses a python type (tuple)
    # and thus can't release the gil
    # using a c data structure (e.g. struct) would overcome gil limitation
    cdef tuple ji_from_n(self, DTYPE_int_t n):
        cdef DTYPE_int_t j, i
        cdef tuple coord
        j = n / self.Nx
        i = n % self.Nx
        coord = (j, i)
        return coord

    cdef DTYPE_int_t n_from_ji(self, DTYPE_int_t j, DTYPE_int_t i) nogil:
        cdef DTYPE_int_t n
        n = i + j*self.Nx
        return n

    cdef bint is_border_n(self, DTYPE_int_t n) nogil:
        cdef DTYPE_int_t j, i
        j = n / self.Nx
        i = n % self.Nx
        return self.is_border_ji(j, i)

    cdef bint is_border_ji(self, DTYPE_int_t j, DTYPE_int_t i) nogil:
        return (i==0) or (i==self.Nx-1) or (j==0) or (j==self.Ny-1)

    cdef int*  _neighbors(self, DTYPE_int_t n) nogil:
        """Given index n, return neighbor indices.

        PARAMETERS
        ----------
        n : int
            1D index of point

        RETURNS
        -------
        nbr : int*
            pointer to array of 6 neighbor indices. Must be freed manually?
        """

        cdef DTYPE_int_t j, i
        cdef bint evenrow
        cdef int* nbr
        nbr = <int*> malloc(sizeof(int) * 6)
        nbr[0] = INT_NOT_FOUND

        j = n / self.Nx
        i = n % self.Nx
        evenrow = j % 2

        # don't even bother with border points
        if self.is_border_ji(j, i):
            return nbr

        if not evenrow:
            nbr[0] = self.n_from_ji(j-1, i)
            nbr[1] = self.n_from_ji(j-1, i+1)
            nbr[2] = self.n_from_ji(j, i+1)
            nbr[3] = self.n_from_ji(j+1, i+1)
            nbr[4] = self.n_from_ji(j+1, i)
            nbr[5] = self.n_from_ji(j, i-1)
        else:
            nbr[0] = self.n_from_ji(j-1, i-1)
            nbr[1] = self.n_from_ji(j-1, i)
            nbr[2] = self.n_from_ji(j, i+1)
            nbr[3] = self.n_from_ji(j+1, i)
            nbr[4] = self.n_from_ji(j+1, i-1)
            nbr[5] = self.n_from_ji(j, i-1)

        return nbr

    def neighbors(self, n):
        """Given index n, return neighbor indices.

        PARAMETERS
        ----------
        n : int
            1D index of point

        RETURNS
        -------
        nbr : arraylike
            Numpy ndarray of neighbor points
        """
        cdef int * nptr = self._neighbors(n)
        cdef int [:] nbr
        if nptr[0] == INT_NOT_FOUND:
            free(nptr)
            return np.array([], DTYPE_int)
        else:
            nbr = <int[:6]> nptr
            numpy_array = np.asarray(nbr.copy())
            free(nptr)
            return numpy_array

    def classify_critical_points(self, np.ndarray[DTYPE_flt_t, ndim=2] a):
        """Identify and classify the critical points of array ``a``.
        0: regular point
        +1: maximum
        -1: minimum
        +2: saddle point
        -2: zero gradient detected
        -3: monkey point

        PARAMETERS
        ----------
        a : arraylike
            two-dimensional hexagonally tessalted field, dtype=float64, shape=2

        RETURNS
        -------
        c : arraylike
            array with the same shape as a, with critical points marked
        """

        # make sure array is correct size
        if (a.shape[0] != self.Ny) or (a.shape[1] != self.Nx):
            raise ValueError('array a is the wrong shape')

        # a raveled view
        cdef DTYPE_flt_t [:] ar
        ar = a.ravel()
        cdef DTYPE_int_t [:] c
        c = np.zeros(self.N, DTYPE_int)

        # loop indices
        cdef int n, k, kprev
        # neighbor pointer
        cdef int* nbr
        # raw difference
        cdef DTYPE_flt_t diff
        # sign of differences
        cdef DTYPE_int_t [:] sign_diff = np.zeros(6, DTYPE_int)
        cdef DTYPE_int_t sum_sign_diff

        for n in range(self.N):
            if not self.is_border_n(n):
                nbr = self._neighbors(n)
                # fill in the differnces
                for k in range(6):
                    diff = ar[n] - ar[nbr[k]]
                    if diff==0.0:
                        sign_diff[k] = 0
                    elif diff>0:
                        sign_diff[k] = 1
                    else:
                        sign_diff[k] = -1

                # loop through again and check signs
                sum_sign_diff = 0
                for k in range(6):
                    # check for zero gradient
                    if sign_diff[k]==0:
                        sum_sign_diff = -2
                        break
                    kprev = (k-1) % 6
                    sum_sign_diff += (sign_diff[k] != sign_diff[kprev])
                if sum_sign_diff==0:
                    # extremum
                    c[n] = sign_diff[0]
                elif sum_sign_diff==2:
                    # regular point
                    #c[n] = 0 # already should be zero
                    pass
                elif sum_sign_diff==-2:
                    # zero gradient point
                    c[n] = -2
                elif sum_sign_diff==4:
                    # saddle
                    c[n] = 2
                else:
                    c[n] = -3
                # overwrite for debugging
                #c[n] = sum_sign_diff
                free(nbr)
        res = np.asarray(c).reshape(self.Ny, self.Nx)
        return res

    cpdef np.ndarray[int, ndim=1] maxima(
              self, np.ndarray[DTYPE_flt_t, ndim=2] a):
        cpoints = self.classify_critical_points(a)
        return np.nonzero(cpoints.ravel()==1)[0]

cdef class HexArrayRegion:

    # the parent region
    cdef HexArray ha
    # the points in the region
    cdef unordered_set[int] members

    def __cinit__(self, HexArray ha):
        self.ha = ha
        #self.members = new unordered_set(x0, y0, x1, y1)

    #def __dealloc__(self):
        #del self.thisptr

    cdef void add_point(self, int pt) nogil:
        self.members.insert(pt)

    cdef unordered_set[int] get_boundary(self) nogil:
        cdef unordered_set[int] boundary
        cdef int n, k
        cdef int* nbr
        cdef int npt
        cdef size_t cnt
        for n in self.members:
            nbr = self.ha._neighbors(n)
            if not nbr[0] == INT_NOT_FOUND:
                for k in range(6):
                    cnt = self.members.count(nbr[k])
                    if cnt==0:
                        boundary.insert(nbr[k])
            free(nbr)
        return boundary

def find_convex_regions(np.ndarray[DTYPE_flt_t, ndim=2] a, int minsize=0):
    """Find convex regions around the extrema of ``a``.

    PARAMETERS
    ----------
    a : arraylike
        The 2D field in which to search for convex regions. Must be dtype=f64.
    minsize : int
        The minimum size of regions to return (number of points)

    RETURNS
    -------
    regions : list of HexArrayRegion elements
    """
