import argparse
from muri.kantan import Scaler

parser = argparse.ArgumentParser(description='Scale images with CPU')
parser.add_argument('--input', '-i', default='images/small.png')
parser.add_argument('--output', '-o', default='./')
parser.add_argument('--gpu', '-g', type=int, default=-1)

arguments = parser.parse_args()
print(arguments.gpu)

def go(arguments):
    Scaler.go(arguments.input, arguments.output)

go(arguments)
