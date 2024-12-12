from os import walk, path
from random import sample
from math import ceil
from argparse import ArgumentParser

# Initialize parser
parser = ArgumentParser()
# Adding optional argument
parser.add_argument("-n", "--DirectoryName",
                    help="The name of the directory to pruned")


parser.add_argument("-k", "--keepProportion",
                    type=int,
                    help="Percentage to keep",
                    default=100)

parser.add_argument("-p", "--Path",
                    help="Global path to directory containing Data and MMR directories")

# Read arguments from command line
args = parser.parse_args()

root = f"{args.Path}/Data/AeBAD_fewer_shot/{args.DirectoryName}"
pattern = '.png'

proportionToDelete = 1.0 - (args.keepProportion / 100)

for pathStr, _, files in walk(path.join(root, 'AeBAD_S', 'train')):
    filesToRemove = sample(sorted(map(lambda filename: path.join(pathStr, filename),
                                      filter(lambda filename: filename[-len(pattern):] == pattern,
                                             files))),
                           ceil(proportionToDelete * len(files)))

    if filesToRemove:
        for f in filesToRemove:
            print(f"{f}")
