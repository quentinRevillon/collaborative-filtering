"""Chargement du code de la branche main depuis une copie locale (worktree ou clone).

Le `src/` de main et le nôtre portent le même nom de paquet : on charge donc les fichiers de
main par leur chemin, sous un autre nom de module, pour éviter tout mélange.

Pour obtenir une copie de main à côté du dépôt :
    git worktree add ../a1-main origin/main
"""
import argparse
import importlib.util
from functools import partial
from pathlib import Path

import myfm.base
from tqdm import tqdm

DEFAULT_MAIN = Path(__file__).resolve().parents[3] / "a1-main"
myfm.base.tqdm = partial(tqdm, disable=True)   # pas de barre de progression myfm dans les logs


def main_path_from_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--main", type=Path, default=DEFAULT_MAIN,
                        help=f"copie locale de la branche main (défaut : {DEFAULT_MAIN})")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if not (args.main / "src").is_dir():
        parser.error(f"pas de dossier src/ dans {args.main} : créer une copie de main avec "
                     "`git worktree add ../a1-main origin/main`, ou passer --main <chemin>")
    return args


def load_main_module(main_dir, name):
    """Charge src/<name>.py de la copie de main, sous le nom de module main_<name>."""
    spec = importlib.util.spec_from_file_location(f"main_{name}", Path(main_dir) / "src" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def coo(R):
    """Notes observées de R au format (u, i, r) utilisé par le code de main."""
    import numpy as np
    u, i = np.nonzero(~np.isnan(R))
    return u, i, R[u, i]
