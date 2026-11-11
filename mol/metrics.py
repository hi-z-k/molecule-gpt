"""MOSES-style generation metrics (validity / uniqueness / novelty) and a grid image."""
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED

RDLogger.DisableLog("rdApp.*")


def score(generated, train_smiles=None):
    """Metrics over a list of generated SMILES strings.

    validity = parses / n, uniqueness = distinct canonical / valid,
    novelty = valid canonical not in train_smiles / valid (None if no training set given).
    """
    n = len(generated)
    canon = []                                  # canonical SMILES of the molecules that parse
    qed = logp = mw = rings = 0.0
    for s in generated:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        canon.append(Chem.MolToSmiles(mol))
        qed += QED.qed(mol)
        logp += Crippen.MolLogP(mol)
        mw += Descriptors.MolWt(mol)
        rings += mol.GetRingInfo().NumRings()
    n_valid = len(canon)
    uniq = set(canon)
    out = {"n": n,
           "validity": n_valid / n if n else 0.0,
           "uniqueness": len(uniq) / n_valid if n_valid else 0.0,
           "novelty": None,
           "qed": qed / n_valid if n_valid else None,
           "logp": logp / n_valid if n_valid else None,
           "mol_weight": mw / n_valid if n_valid else None,
           "rings": rings / n_valid if n_valid else None,
           "valid_canonical": canon}
    if train_smiles is not None:
        known = train_smiles if isinstance(train_smiles, (set, frozenset)) else set(train_smiles)
        out["novelty"] = sum(c not in known for c in canon) / n_valid if n_valid else 0.0
    return out


def draw_grid(smiles, path, n=12):
    """Save a grid image of the first n molecules; False if RDKit cannot draw here."""
    try:
        from rdkit.Chem import Draw
        mols = [m for m in (Chem.MolFromSmiles(s) for s in smiles[:n]) if m is not None]
        if not mols:
            return False
        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(250, 250), useSVG=False)
        data = getattr(img, "data", None)       # IPython image in notebooks, PIL image otherwise
        if data is not None:
            with open(path, "wb") as f:
                f.write(data)
        else:
            img.save(path)
        return True
    except Exception as e:
        print(f"could not draw grid: {e}")
        return False
