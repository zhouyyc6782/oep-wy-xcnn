from __future__ import print_function
import sys
from model import CNN_GGA_1
from func import load_model, save_model

print(sys.argv)

m = CNN_GGA_1()

load_model(m, sys.argv[1])
m.cpu()
m.cpu()
save_model(m, sys.argv[2])

