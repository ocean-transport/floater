"""For generating grids of floats."""

import numpy as np

class FloatSet(object):
    """Represents a set of initial float positions on a regular grid."""

    def __init__(self, xlim, ylim, dx=1., dy=1.):
        """Initialize FloatSet according to specified rectangular grid geometry.

        PARAMETERS
        ----------
        xlim : tuple of two floats
            minimum and maximum x coordinate of cell vertices
        ylim : tuple of two floats
            minimum and maximum y coordinate of cell vertices
        dx : float
            grid spacing in x direction
        dy : float
            grid spacing in y direction
        """

        self.xlim = xlim
        self.ylim = ylim
        self.Lx = xlim[1] - xlim[0]
        self.Ly = ylim[1] - ylim[0]
        if (self.Lx % dx) != 0.0:
            raise ValueError("Lx is not divisible evenly by dx")
        if (self.Ly % dy) != 0.0:
            raise ValueError("Ly is not divisible evenly by dy")
        self.Nx = int(self.Lx / dx)
        self.Ny = int(self.Ly / dy)
        self.dx = dx
        self.dy = dy
        self.x = self.xlim[0] + self.dx * np.arange(self.Nx) + self.dx/2
        self.y = self.ylim[0] + self.dy * np.arange(self.Ny) + self.dy/2

    def get_rectmesh(self):
        """Get the coordinates of the float positions in a rectangualr mesh.

        RETURNS
        -------
        x : np.ndarray
            2D array of float x coordinates
        y : np.ndarray
            2D array of float y coordinates
        """

        return np.meshgrid(self.x, self.y)

    def get_hexmesh(self):
        """Get the coordinates of the float positions in a hexagonal mesh.

        RETURNS
        -------
        x : np.ndarray
            2D array of float x coordinates
        y : np.ndarray
            2D array of float y coordinates
        """

        xx, yy = self.get_rectmesh()
        # modify to be even-R horizontal offset
        xx[::2] += self.dx/4
        xx[1::2] -= self.dx/4

        return xx, yy
