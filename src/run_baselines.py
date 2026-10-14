"""Compare les baselines moyenne / biais sur le split local train -> test.

Protocole :
  1. les lambdas sont choisis sur une validation (10 % des notes de train) ;
  2. chaque modèle est ensuite réentraîné sur tout le train et évalué sur le test.

Lancer depuis la racine du repo :  python -m src.run_baselines
"""
import time

from src.baselines import Biases, GlobalMean
from src.data import load_train_test, split_observed
from src.metrics import accuracy, rmse

LAMBDAS = [0, 1, 2, 5]


def tune_lambda(R_fit, R_val, make_model):
    scores = {lam: rmse(R_val, make_model(lam).fit(R_fit).predict()) for lam in LAMBDAS}
    best = min(scores, key=scores.get)
    print("    val RMSE : " + "  ".join(f"lam={lam}:{s:.4f}" for lam, s in scores.items()))
    return best


def main():
    R_train, R_test = load_train_test()
    R_fit, R_val = split_observed(R_train, frac=0.1, seed=0)

    print("Reglage des lambdas sur la validation :")
    print("  Biais film seul")
    lam_item = tune_lambda(R_fit, R_val, lambda l: Biases(lam_i=l, use_user=False))
    print("  Biais user seul")
    lam_user = tune_lambda(R_fit, R_val, lambda l: Biases(lam_u=l, use_item=False))
    print("  Biais user + film (lam_u = lam_i)")
    lam_both = tune_lambda(R_fit, R_val, lambda l: Biases(lam_u=l, lam_i=l))

    models = {
        "Moyenne globale": GlobalMean(),
        "Biais film (mu + b_i)": Biases(use_user=False),
        "Biais user (mu + b_u)": Biases(use_item=False),
        "Biais user+film, sequentiel": Biases(n_iter=1),
        "Biais user+film, moindres carres": Biases(n_iter=20),
        f"Biais film regularise (lam={lam_item})": Biases(lam_i=lam_item, use_user=False),
        f"Biais user regularise (lam={lam_user})": Biases(lam_u=lam_user, use_item=False),
        f"Biais user+film regularises (lam={lam_both})": Biases(lam_u=lam_both, lam_i=lam_both),
    }

    print(f"\n{'Modele':<45}{'RMSE train':>11}{'RMSE test':>11}{'Acc test':>10}{'Temps':>9}")
    for name, model in models.items():
        t0 = time.perf_counter()
        P = model.fit(R_train).predict()
        dt = time.perf_counter() - t0
        print(f"{name:<45}{rmse(R_train, P):>11.4f}{rmse(R_test, P):>11.4f}"
              f"{accuracy(R_test, P):>10.3f}{dt:>8.2f}s")


if __name__ == "__main__":
    main()
