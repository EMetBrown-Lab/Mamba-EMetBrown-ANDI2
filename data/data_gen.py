from andi_datasets.datasets_challenge import challenge_phenom_dataset
from multiprocessing import Pool
import numpy as np 

def main(n):
    ## Important otherwise the seed
    np.random.seed()

    challenge_phenom_dataset(
        experiments = 10000,
        save_data=True,
        files_reorg = True,
        repeat_exp = True,
        path = "data/",
        save_labels_reorg = True,
        prefix = str(n) + "_",
        delete_raw = True,
        )

# main(1)
# main(0)
if __name__ == '__main__':
    with Pool(18) as p:
        print(p.map(main, range(101,1000)))