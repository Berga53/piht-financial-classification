"""Bankitalia panel preparation extracted from the final-chapter notebook."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


AUTONOMOUS_REGIONS = {
    "VALLE D'AOSTA/VALLÉE D'AOSTE",
    "TRENTINO-ALTO ADIGE/SÜDTIROL",
    "FRIULI VENEZIA GIULIA",
    "SARDEGNA",
}


@dataclass
class PreparedDataset:
    X: NDArray[np.float64]
    y: NDArray[np.int64]
    groups: NDArray
    window_start: NDArray[np.int64]
    feature_names: list[str]
    static_feature_count: int
    metadata: dict


def source_fingerprint(root: Path) -> str:
    """Hash the CSV inputs so resume cannot confuse changed data with an old run."""
    digest = hashlib.sha256()
    for path in sorted((root / "data").rglob("*.csv")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _read_yearly(root: Path, directory: str, years: range) -> dict[int, pd.DataFrame]:
    return {
        year: pd.read_csv(root / "data" / directory / f"{directory} {year}.csv", index_col="BDAP")
        for year in years
    }


def _targets(comuni: pd.DataFrame, root: Path) -> pd.DataFrame:
    result = comuni.copy()
    critici = pd.read_csv(
        root / "data" / "CriticitàComuni.csv", sep=";", encoding="latin_1", low_memory=False
    )
    for year in range(2000, 2025):
        result[f"Target {year}"] = 0
    for row in critici.itertuples(index=False):
        if pd.isna(row.Anno) or pd.isna(row.Comune):
            continue
        year = int(row.Anno)
        if not 2000 <= year <= 2024:
            continue
        municipality = str(row.Comune)
        alternative = municipality.replace("-", " ")
        matches = result["Comune"].isin(
            (f"COMUNE DI {municipality}", f"COMUNE DI {alternative}")
        )
        result.loc[matches, f"Target {year}"] = 1
    return result


def _window_starts(period: int, input_depth: int, target_depth: int) -> list[int]:
    if period == 1:
        # Exact definition used in classification.ipynb.
        return list(range(2009, 2015 - input_depth + 2))
    if period == 2:
        return list(range(2016, 2023 - input_depth - target_depth + 3))
    raise ValueError("period must be 1 or 2")


def prepare_bankit_dataset(
    bankit_root: str | Path,
    *,
    period: int,
    input_depth: int,
    target_depth: int,
    features: str = "bdap",
    exclude_autonomous_regions: bool = False,
) -> PreparedDataset:
    root = Path(bankit_root).expanduser().resolve()
    valid_features = {
        "anticipazioni",
        "bdap",
        "bdap-anticipazioni",
        "bdap-anticipazioni-reduced",
        "bdap-indicatori-anticipazioni",
        "indicatori",
        "indicatori-anticipazioni",
        "readybdap-anticipazioni",
    }
    if features not in valid_features:
        raise ValueError(f"features must be one of {sorted(valid_features)}")
    if input_depth < 1 or target_depth < 1:
        raise ValueError("input_depth and target_depth must be positive")

    years = range(2009, 2024)
    comuni = pd.read_csv(root / "data" / "comuni.csv", index_col="BDAP")
    comuni = _targets(comuni, root)
    if exclude_autonomous_regions:
        comuni = comuni.loc[~comuni["Regione"].isin(AUTONOMOUS_REGIONS)]

    use_bdap = features.startswith("bdap") or features.startswith("readybdap")
    use_anticipazioni = "anticipazioni" in features
    use_indicatori = "indicatori" in features
    use_reduced_bdap = features in {
        "bdap-anticipazioni-reduced",
        "readybdap-anticipazioni",
    }

    spese = _read_yearly(root, "Spese", years) if use_bdap else None
    entrate = _read_yearly(root, "Entrate", years) if use_bdap else None
    if use_reduced_bdap:
        spese = {
            year: frame[
                [column for column in frame.columns if "Impegno" in column or "Impegni" in column]
            ]
            for year, frame in spese.items()
        }
        entrate = {
            year: frame[
                [
                    column
                    for column in frame.columns
                    if "Accertamento" in column or "Accertamenti" in column
                ]
            ]
            for year, frame in entrate.items()
        }
    anticipazioni = _read_yearly(root, "Anticipazioni", years) if use_anticipazioni else None
    indicatori = _read_yearly(root, "Indicatori", range(2016, 2024)) if use_indicatori else None

    if use_indicatori:
        if period != 2:
            raise ValueError("indicatori are only available for period 2")
        # Keep the common national roster; missing indicator records are imputed to zero.
        indicatori = {year: table.reindex(comuni.index).fillna(0.0)
                      for year, table in indicatori.items()}

    population = pd.read_csv(root / "data" / "popolazione.csv", index_col="BDAP")
    zones = pd.read_csv(root / "data" / "zona.csv", index_col="BDAP") if use_bdap else None
    starts = _window_starts(period, input_depth, target_depth)
    if not starts:
        raise ValueError("the requested input/target depths produce no time windows")

    feature_tables = [table for table in (spese, entrate, anticipazioni, indicatori) if table]
    first_year = starts[0]
    feature_names: list[str] = []
    for table in feature_tables:
        feature_names.extend(map(str, table[first_year].columns))
    if use_bdap:
        feature_names.append("log_population")
    static_feature_count = 0
    if zones is not None:
        feature_names.extend(map(str, zones.columns))
        static_feature_count = zones.shape[1]

    X_windows: list[NDArray[np.float64]] = []
    y_windows: list[NDArray[np.int64]] = []
    group_windows: list[NDArray] = []
    start_windows: list[NDArray[np.int64]] = []

    for start in starts:
        yearly_arrays = []
        for year in range(start, start + input_depth):
            columns = []
            for table in feature_tables:
                columns.append(table[year].reindex(comuni.index))
            if use_bdap:
                logged_population = np.log(population[str(year)].reindex(comuni.index))
                columns.append(logged_population.rename("log_population").to_frame())
            if zones is not None:
                columns.append(zones.reindex(comuni.index))
            frame = pd.concat(columns, axis=1)
            yearly_arrays.append(frame.to_numpy(dtype=float))

        panel = np.stack(yearly_arrays, axis=1)
        target_sum = sum(
            comuni[f"Target {start + input_depth - 1 + horizon}"].to_numpy()
            for horizon in range(1, target_depth + 1)
        )
        X_windows.append(panel)
        y_windows.append((target_sum > 0).astype(int))
        group_windows.append(comuni.index.to_numpy())
        start_windows.append(np.full(len(comuni), start, dtype=int))

    X = np.concatenate(X_windows)
    if not np.isfinite(X).all():
        bad = int(np.size(X) - np.isfinite(X).sum())
        raise ValueError(
            f"prepared data contain {bad} missing or infinite values; clean/impute them before fitting"
        )

    return PreparedDataset(
        X=X,
        y=np.concatenate(y_windows).astype(int),
        groups=np.concatenate(group_windows),
        window_start=np.concatenate(start_windows),
        feature_names=feature_names,
        static_feature_count=static_feature_count,
        metadata={
            "bankit_root": str(root),
            "period": period,
            "input_depth": input_depth,
            "target_depth": target_depth,
            "features": features,
            "excluded_autonomous_regions": exclude_autonomous_regions,
            "municipality_count": len(comuni),
            "source_sha256": source_fingerprint(root),
            "indicator_cohort": "common_roster",
            "indicator_missing": "zero",
            "data_protocol": "national_common_roster_v1",
        },
    )


def save_dataset(dataset: PreparedDataset, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=dataset.X,
        y=dataset.y,
        groups=dataset.groups,
        window_start=dataset.window_start,
        feature_names=np.asarray(dataset.feature_names, dtype=str),
        static_feature_count=np.asarray(dataset.static_feature_count),
        metadata=np.asarray(json.dumps(dataset.metadata)),
    )


def load_dataset(path: str | Path) -> PreparedDataset:
    with np.load(path, allow_pickle=False) as archive:
        return PreparedDataset(
            X=archive["X"],
            y=archive["y"],
            groups=archive["groups"],
            window_start=archive["window_start"],
            feature_names=archive["feature_names"].tolist(),
            static_feature_count=int(archive["static_feature_count"]),
            metadata=json.loads(str(archive["metadata"])),
        )


def flatten_panel(dataset: PreparedDataset) -> tuple[NDArray[np.float64], list[str]]:
    X = dataset.X
    if X.ndim != 3:
        raise ValueError(f"expected a sample x time x feature tensor, got shape {X.shape}")
    n_samples, depth, n_features = X.shape
    n_static = dataset.static_feature_count
    n_temporal = n_features - n_static
    temporal = X[:, :, :n_temporal].reshape(n_samples, depth * n_temporal)
    names = [
        f"{name}@t-{depth - step - 1}"
        for step in range(depth)
        for name in dataset.feature_names[:n_temporal]
    ]
    if n_static:
        temporal = np.concatenate((temporal, X[:, 0, n_temporal:]), axis=1)
        names.extend(dataset.feature_names[n_temporal:])
    return temporal, names
