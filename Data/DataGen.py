from nnfs.datasets import spiral_data
import nnfs

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

