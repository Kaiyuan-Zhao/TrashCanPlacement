import numpy
import pandas
import matplotlib.pyplot as plt


# building layout class

class BuildingLayout:
    # array of zeros with dimensions rows x cols
    # free space = 0, wall = 1
    def __init__(self, rows, cols, layout=None):
        self.rows = rows
        self.cols = cols
        self.layout = numpy.zeros((rows, cols))
        if layout is not None:
            self.layout = layout


openRoom = BuildingLayout(10, 10, [])


