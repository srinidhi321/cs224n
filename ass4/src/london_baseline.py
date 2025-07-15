# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import utils
from tqdm import tqdm



def main():
    accuracy = 0.0
    count = 0
    for line in tqdm(open(args.eval_corpus_path, encoding='utf-8')):
          x = line.split('\t')[1]
          if x == 'London\n':
            accuracy += 1
          count += 1
    accuracy = accuracy / count
    return 100*accuracy

if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--eval_corpus_path', default=None)
    args = argp.parse_args()
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")


# python src/london_baseline.py --eval_corpus_path birth_dev.tsv