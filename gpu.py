import argparse
import maji

parser = argparse.ArgumentParser(description='無駄な事')
parser.add_argument('--input', '-i', default='images/small.png')
parser.add_argument('--output', '-o', default='./')
parser.add_argument('--gpu', '-g', type=int, default=0)

arguments = parser.parse_args()
print(arguments.gpu)

def go(arguments):
    maji.Scaler.go(arguments.input, arguments.output)

go(arguments)
