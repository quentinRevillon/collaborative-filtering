"""Protocole final et sauvegarde des résultats.

Les modèles (hyperparamètres déjà choisis par CV, cf. src.tuning) sont réentraînés sur
tout le train puis scorés une seule fois sur le test.
"""
import csv
import time
from pathlib import Path

from src.metrics import accuracy, rmse

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def evaluate_on_test(models, R_train, R_test, csv_name=None):
    """models : {nom: modèle non entraîné}. Affiche le tableau et l'écrit dans results/."""
    rows = []
    print(f"\n{'Modele':<50}{'RMSE train':>11}{'RMSE test':>11}{'Acc test':>10}{'Temps':>9}")
    for name, model in models.items():
        t0 = time.perf_counter()
        R_hat = model.fit(R_train).predict()
        dt = time.perf_counter() - t0
        row = dict(model=name, train_rmse=round(rmse(R_train, R_hat), 4),
                   test_rmse=round(rmse(R_test, R_hat), 4),
                   test_acc=round(accuracy(R_test, R_hat), 3), seconds=round(dt, 2))
        rows.append(row)
        print(f"{name:<50}{row['train_rmse']:>11.4f}{row['test_rmse']:>11.4f}"
              f"{row['test_acc']:>10.3f}{dt:>8.2f}s")
    if csv_name:
        write_csv(csv_name, rows)
    return rows


def write_csv(name, rows):
    """Écrit des lignes (dicts, colonnes éventuellement différentes) dans results/<name>."""
    RESULTS_DIR.mkdir(exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(RESULTS_DIR / name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(name):
    """Relit un CSV de results/ en convertissant les nombres et booléens."""
    with open(RESULTS_DIR / name, newline="") as f:
        return [{k: _parse(v) for k, v in row.items()} for row in csv.DictReader(f)]


def best_row(name, variant=None):
    """Meilleure ligne (cv_rmse minimal) d'une grille sauvegardée, éventuellement pour une
    seule variante. Renvoie None si le fichier n'existe pas encore."""
    if not (RESULTS_DIR / name).exists():
        return None
    rows = [r for r in read_csv(name) if variant is None or r.get("variant") == variant]
    return min(rows, key=lambda r: r["cv_rmse"])


def _parse(value):
    if value in ("", None):
        return None
    if value in ("True", "False"):
        return value == "True"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value
