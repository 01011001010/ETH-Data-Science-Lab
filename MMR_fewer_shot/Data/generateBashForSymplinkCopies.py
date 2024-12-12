from argparse import ArgumentParser


# Initialize parser
parser = ArgumentParser()
# Adding optional argument
parser.add_argument("-n", "--NewDirectoryName",
                    help="The name of the new directory to be generated")
parser.add_argument("-p", "--Path",
                    help="Global path to directory containing Data and MMR directories")
# Read arguments from command line
args = parser.parse_args()


newRoot = args.NewDirectoryName
basePath = f"{args.Path}/Data/AeBAD_fewer_shot/" #todo
rootOriginalPath = f"{args.Path}/Data/AeBAD/"#todo


print(f"#!/bin/bash\n\nmkdir {basePath}{newRoot}\n")


alreadyCreated = set()
for fullPath in map(lambda p: p.split('/'),
                    (#'AeBAD_S/ground_truth/ablation/background',
                     # 'AeBAD_S/ground_truth/ablation/illumination',
                     # 'AeBAD_S/ground_truth/ablation/same',
                     # 'AeBAD_S/ground_truth/ablation/view',
                     #
                     # 'AeBAD_S/ground_truth/breakdown/background',
                     # 'AeBAD_S/ground_truth/breakdown/illumination',
                     # 'AeBAD_S/ground_truth/breakdown/same',
                     # 'AeBAD_S/ground_truth/breakdown/view',
                     #
                     # 'AeBAD_S/ground_truth/fracture/background',
                     # 'AeBAD_S/ground_truth/fracture/illumination',
                     # 'AeBAD_S/ground_truth/fracture/same',
                     # 'AeBAD_S/ground_truth/fracture/view',
                     #
                     # 'AeBAD_S/ground_truth/groove/background',
                     # 'AeBAD_S/ground_truth/groove/illumination',
                     # 'AeBAD_S/ground_truth/groove/same',
                     # 'AeBAD_S/ground_truth/groove/view',
                     #
                     #
                     # 'AeBAD_S/test/ablation/background',
                     # 'AeBAD_S/test/ablation/illumination',
                     # 'AeBAD_S/test/ablation/same',
                     # 'AeBAD_S/test/ablation/view',
                     #
                     # 'AeBAD_S/test/breakdown/background',
                     # 'AeBAD_S/test/breakdown/illumination',
                     # 'AeBAD_S/test/breakdown/same',
                     # 'AeBAD_S/test/breakdown/view',
                     #
                     # 'AeBAD_S/test/fracture/background',
                     # 'AeBAD_S/test/fracture/illumination',
                     # 'AeBAD_S/test/fracture/same',
                     # 'AeBAD_S/test/fracture/view',
                     #
                     # 'AeBAD_S/test/good/background',
                     # 'AeBAD_S/test/good/illumination',
                     # 'AeBAD_S/test/good/same',
                     # 'AeBAD_S/test/good/view',
                     #
                     # 'AeBAD_S/test/groove/background',
                     # 'AeBAD_S/test/groove/illumination',
                     # 'AeBAD_S/test/groove/same',
                     # 'AeBAD_S/test/groove/view',

                     'AeBAD_S/train/good/background',
                     'AeBAD_S/train/good/illumination',
                     'AeBAD_S/train/good/view'
                     )):
    for i in range(len(fullPath)):
        if '/'.join(fullPath[:i+1]) in alreadyCreated:
            continue
        print(f"mkdir {basePath}{newRoot}/{'/'.join(fullPath[:i+1])}")
        alreadyCreated.add('/'.join(fullPath[:i+1]))
    print(f"cd {basePath}{newRoot}/{'/'.join(fullPath)}")
    print(f"ln -s {rootOriginalPath}{'/'.join(fullPath)}/*.png .")
    print('\n')


print(f"ln -s {rootOriginalPath}AeBAD_S/test {basePath}{newRoot}/AeBAD_S/test")
print(f"ln -s {rootOriginalPath}AeBAD_S/ground_truth {basePath}{newRoot}/AeBAD_S/ground_truth")
