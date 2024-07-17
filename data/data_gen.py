from andi_datasets.datasets_challenge import (
    challenge_phenom_dataset,
    _get_dic_andi2,
)

from scipy.stats import loguniform
from multiprocessing import Pool
import numpy as np


def generate_random_experiment(n_exp, m=None):
    if m is None:
        models = np.random.choice([1, 2, 3, 4, 5], size=n_exp)
    else:
        models = [m] * n_exp

    dics = []

    for m in models:
        dic = _get_dic_andi2(m)

        dic["T"] = 200
        dic["N"] = np.random.randint(low=20, high=150, dtype=int)
        dic["L"] = 1.5 * 128

        K_1 = loguniform.rvs(1e-12, 1e6)
        sigma_K_1 = loguniform.rvs(0.001, 1)


        alpha_1 = loguniform.rvs(0.01, 2)
        sigma_alpha_1 = loguniform.rvs(0.001, 1)

        while K_1 * (20**alpha_1) > 1000:
                print(f"rejected {K_1} {alpha_1}")
                K_1 = loguniform.rvs(1e-12, 1e6)
                alpha_1 = np.random.uniform(0, 2)
                


        # Particle / trap binding and unbinding probabilities for dimerization / immobilization
        if m in [3, 4]:
            dic.update(
                {
                    "Pu": loguniform.rvs(0.01, 0.4),
                    "Pb": loguniform.rvs(0.01, 1),
                    "r": np.random.uniform(0.3, 0.7),
                }
            )

        ## Single state

        if m == 1:
            

            dic.update(
                {
                    "Ds": np.array([K_1, sigma_K_1]),
                    "alphas": np.array([alpha_1, sigma_alpha_1]),
                }
            )

        ## Multi state

        if m == 2:

            # while K_1 * (20**alpha_1) > 128:
            #     K_1 = loguniform.rvs(1e-12, 1e6)
            #     alpha_1 = np.random.uniform(0, 2)

            if K_1 > 1e-11:
                K_2 = K_1 * loguniform.rvs(0.001, 0.2)
            else:
                K_2 = K_1 * loguniform.rvs(4, 100)

            sigma_K_2 = np.random.randn() * (10 * sigma_K_1) + sigma_K_1
            while sigma_K_2 < 1e-12 or sigma_K_2 > 1e6:
                sigma_K_2 = np.random.randn() * (10 * sigma_K_1) + sigma_K_1

            alpha_2 = alpha_1 - (np.random.rand() * (0.6 - 0.2) + 0.2)
            while np.abs(alpha_2 - alpha_1) < 0.2:
                if alpha_1 > 0.2:
                    alpha_2 = alpha_1 - (np.random.rand() * (0.6 - 0.2) + 0.2)
                else:
                    alpha_2 = alpha_1 + (np.random.rand() * (0.6 - 0.2) + 0.2)

            sigma_alpha_2 = np.random.randn() * (0.1 * sigma_alpha_1) + sigma_alpha_1

            dic.update(
                {
                    "Ds": np.array(
                        [[K_1, sigma_K_1], [K_2, sigma_K_2]],
                    ),
                    "alphas": np.array(
                        [[alpha_1, sigma_alpha_1], [alpha_2, sigma_alpha_2]],
                    ),
                }
            )

            P1 = np.random.uniform(0.85, 0.98)
            P2 = np.random.uniform(0.85, 0.98)

            M = [[P1, 1 - P1], [P2, 1 - P2]]

            dic["M"] = M

        ## Traps
        if m == 3:
            # number of traps
            dic.update({"Nt": np.random.randint(low=250, high=350)})

            trap_mean_distance = 1 / (dic["Nt"] / 128**2) ** (
                1 / 3
            )  # Approximation for mean dist between particles

            while (
                K_1 * (20**alpha_1) > (trap_mean_distance) ** 2
            ):  # Is the particle potentially jumping from trap to trap ? is it jumping out of the trap ?
                K_1 = loguniform.rvs(1e-12, 1e6)
                alpha_1 = np.random.uniform(0, 2)

            dic.update(
                {
                    "Ds": np.array([K_1, sigma_K_1]),
                    "alphas": np.array([alpha_1, sigma_alpha_1]),
                }
            )

        ## Dimerization
        if m == 4:
            if K_1 > 1e-11:
                K_2 = K_1 * loguniform.rvs(0.001, 0.2)
            else:
                K_2 = K_1 * loguniform.rvs(4, 100)

            sigma_K_2 = np.random.randn() * (10 * sigma_K_1) + sigma_K_1
            while sigma_K_2 < 1e-12 or sigma_K_2 > 1e6:
                sigma_K_2 = np.random.randn() * (10 * sigma_K_1) + sigma_K_1

            alpha_2 = alpha_1 - (np.random.rand() * (0.6 - 0.2) + 0.2)
            while np.abs(alpha_2 - alpha_1) < 0.2:
                if alpha_1 > 0.2:
                    alpha_2 = alpha_1 - (np.random.rand() * (0.6 - 0.2) + 0.2)
                else:
                    alpha_2 = alpha_1 + (np.random.rand() * (0.6 - 0.2) + 0.2)

            sigma_alpha_2 = np.random.randn() * (0.1 * sigma_alpha_1) + sigma_alpha_1

            sigma_alpha_2 = np.random.randn() * (0.1 * sigma_alpha_1) + sigma_alpha_1

            dic.update(
                {
                    "Ds": np.array(
                        [[K_1, sigma_K_1], [K_2, sigma_K_2]],
                    ),
                    "alphas": np.array(
                        [[alpha_1, sigma_alpha_1], [alpha_2, sigma_alpha_2]],
                    ),
                }
            )

        ## Confinement
        if m == 5:
            dic.update({"r": np.random.uniform(3, 8), "Nc": np.random.randint(25, 35)})

            confinement_mean_distance = 1 / (dic["Nc"] / 128**2) ** (
                1 / 3
            )  # Approximation for mean dist between particles

            while (
                K_1 * (20**alpha_1) > (confinement_mean_distance) ** 2
            ):  # Is the particle potentially jumping from trap to trap ? is it jumping out of the trap ?
                K_1 = loguniform.rvs(1e-12, 1e6)
                alpha_1 = np.random.uniform(0, 2)

            if K_1 > 1e-11:
                K_2 = K_1 * loguniform.rvs(0.001, 0.2)
            else:
                K_2 = K_1 * loguniform.rvs(4, 100)

            sigma_K_2 = np.random.randn() * (10 * sigma_K_1) + sigma_K_1
            alpha_2 = alpha_1 - (np.random.rand() * (0.6 - 0.2) + 0.2)
            while sigma_K_2 < 1e-12 or sigma_K_2 > 1e6:
                sigma_K_2 = np.random.randn() * (10 * sigma_K_1) + sigma_K_1

            while np.abs(alpha_2 - alpha_1) < 0.2:
                if alpha_1 > 0.2:
                    alpha_2 = alpha_1 - (np.random.rand() * (0.6 - 0.2) + 0.2)
                else:
                    alpha_2 = alpha_1 + (np.random.rand() * (0.6 - 0.2) + 0.2)

            sigma_alpha_2 = np.random.randn() * (0.1 * sigma_alpha_1) + sigma_alpha_1

            sigma_alpha_2 = np.random.randn() * (0.1 * sigma_alpha_1) + sigma_alpha_1

            dic.update(
                {
                    "Ds": np.array(
                        [[K_1, sigma_K_1], [K_2, sigma_K_2]],
                    ),
                    "alphas": np.array(
                        [[alpha_1, sigma_alpha_1], [alpha_2, sigma_alpha_2]],
                    ),
                }
            )

        dics.append(dic)

    return dics


# %%
def main(n):
    try: 
        np.random.seed()
        dics = generate_random_experiment(50, m=n[0])
        challenge_phenom_dataset(
            dics=dics,
            save_data=True,
            files_reorg=False,
            repeat_exp=True,
            path= n[2] + str(n[0]) + "/" + str(n[1]) + "/",
            save_labels_reorg=True,
            return_timestep_labs=True,
            # prefix = str(n) + "_",
            delete_raw=True,
        )
    except:
        pass


# main((1,1))
if __name__ == "__main__":

    # for i in range(1, 2):
    #     a = generate_random_experiment(10000, m=i)
    #     print(a)
    #     challenge_phenom_dataset(
    #         dics=a,
    #         save_data=False,
    #         path="./test/",
    #         num_fovs=1,
    #         files_reorg=False,
    #         save_labels_reorg=False,
    #         return_timestep_labs=True,
    #     )
    # for batch in range(3,20):

    #     with Pool(10) as p:
    #         exp = np.arange(20)
    #         exps = np.random.choice(exp, size=1000)
    #         runs = [(i, n, f"./datasets/small_batch_{batch}/") for n, i in enumerate(exps)]
    #         print(p.map(main, runs))

    np.random.seed()
    dics = generate_random_experiment(10)


    challenge_phenom_dataset(
            dics=dics,
            save_data=True,
            files_reorg=False,
            # repeat_exp=True,
            path= "/home/m.lavaud/ANDI_2_Challenge_EMetBrown/data/test_dataset/ref/exps_to_move",
            save_labels_reorg=True,
            return_timestep_labs=True,
            num_fovs = 30,
            # prefix = str(n) + "_",
            # delete_raw=True,
    )

# %%
