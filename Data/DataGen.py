from nnfs.datasets import spiral_data
import nnfs

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

f = open('data.txt', 'w+')

f.write("Start_X:\n")
f.write(f'{X}')
f.write("\nStart_Y:\n")
f.write(f'{y}\n')
f.close()
exit()


