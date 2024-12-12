from argparse import ArgumentParser
from re import match

# Initialize parser
parser = ArgumentParser()
parser.add_argument("-r", "--ReadFile",
                    help="Filename of the run output")

parser.add_argument("-w", "--OutputFile",
                    help="Filename to write filtered to")
# Read arguments from command line
args = parser.parse_args()

with open(args.ReadFile, 'r') as inFile:
    with open(args.OutputFile, 'w') as outFile:
        outFile.write(f"seed,percent of training data,normal shots,domain shift,classification AUROC,pixel level AUROC,per-region-overlap (PRO) AUROC\n")
        seed = percent = k = domainShift = AUROC = PxAUROC = PRO = None
        while line := inFile.readline():
            m = match(r".+main.py:   27.+/(\d+)_percent_(\d+)_seed", line)
            if m:
                percent = m.group(1)
                seed = m.group(2)
                print(m)
                continue
            m = match(r".+train.py:   56[^\d]+(\d+)", line)
            if m:
                k = m.group(1)
                print(m)
                continue
            m = match(r".+train.py:  104.+(same|background|illumination|view)", line)
            if m:
                domainShift = m.group(1)
                print(m)
                continue
            m = match(r".+train.py:  134[^\d]+(\d+\.\d+)", line)
            if m:
                AUROC = m.group(1)
                print(m)
                continue
            m = match(r".+train.py:  137[^\d]+(\d+\.\d+)", line)
            if m:
                PxAUROC = m.group(1)
                print(m)
                continue
            m = match(r".+train.py:  141[^\d]+(\d+\.\d+)", line)
            if m:
                PRO = m.group(1)
                print(m)
                outFile.write(f"{seed},{percent},{k},{domainShift},{AUROC},{PxAUROC},{PRO}\n")
                continue
            m = match(r".+train.py:  147: Mean AUROC[^\d]+(\d+\.\d+)", line)
            if m:
                AUROC = m.group(1)
                print(m)
                domainShift = 'mean'
                continue
            m = match(r".+train.py:  147: Mean Pixel-AUROC[^\d]+(\d+\.\d+)", line)
            if m:
                PxAUROC = m.group(1)
                print(m)
                continue
            m = match(r".+train.py:  147: Mean per-region-overlap[^\d]+(\d+\.\d+)", line)
            if m:
                PRO = m.group(1)
                print(m)
                outFile.write(f"{seed},{percent},{k},{domainShift},{AUROC},{PxAUROC},{PRO}\n")


