"""Display labels matching the feature codes in the thesis variable catalogue."""

from __future__ import annotations


FEATURE_LABELS = {
    "log_population": r"$P$",
    "NORD-OVEST": r"$Z1$",
    "NORD-EST": r"$Z2$",
    "CENTRO": r"$Z3$",
    "SUD": r"$Z4$",
    "ISOLE": r"$Z5$",
    **{f"A{i}": rf"$A{i}$" for i in range(1, 6)},
    **{f"R{i}": rf"$R{i}$" for i in range(1, 11)},
}


_P1_EXPENDITURE = [
    "SPESE CORRENTI",
    "SPESE IN CONTO CAPITALE",
    "SPESE PER RIMBORSO DI PRESTITI",
    "SPESE PER SERVIZI PER CONTO DI TERZI",
]
for index, category in enumerate(_P1_EXPENDITURE, start=1):
    FEATURE_LABELS[f"Impegno totale {category}"] = rf"$S{index}_{{P1}}$ COM"
    FEATURE_LABELS[f"Pagamenti in CC {category}"] = rf"$S{index}_{{P1}}$ PAY CY"
    FEATURE_LABELS[f"Pagamenti in CR {category}"] = rf"$S{index}_{{P1}}$ PAY PY"

_P1_REVENUE = [
    "ENTRATE DA SERVIZI PER CONTO DI TERZI",
    "ENTRATE DERIVANTI DA ACCENSIONE DI PRESTITI",
    "ENTRATE DERIVANTI DA ALIENAZIONE, DA TRASFERIMENTI DI CAPITALE E DA RISCOSSIONI DI CREDITI",
    "ENTRATE DERIVANTI DA CONTRIBUTI E TRASFERIMENTI CORRENTI",
    "ENTRATE EXTRATRIBUTARIE",
    "ENTRATE TRIBUTARIE",
]
for index, category in enumerate(_P1_REVENUE, start=1):
    FEATURE_LABELS[f"Accertamento in CC {category}"] = rf"$E{index}_{{P1}}$ REC"
    FEATURE_LABELS[f"Riscossione in CC {category}"] = rf"$E{index}_{{P1}}$ COL CY"
    FEATURE_LABELS[f"Riscossione in CR {category}"] = rf"$E{index}_{{P1}}$ COL PY"

_P2_EXPENDITURE = [
    "Chiusura Anticipazioni ricevute da istituto tesoriere/cassiere",
    "Rimborso Prestiti",
    "Spese correnti",
    "Spese in conto capitale",
    "Spese per incremento attivita' finanziarie",
    "Uscite per conto terzi e partite di giro",
]
for index, category in enumerate(_P2_EXPENDITURE, start=1):
    FEATURE_LABELS[f"Impegni {category}"] = rf"$S{index}_{{P2}}$ COM"
    FEATURE_LABELS[f"Pagamenti in CC {category}"] = rf"$S{index}_{{P2}}$ PAY CY"
    FEATURE_LABELS[f"Pagamenti in CR {category}"] = rf"$S{index}_{{P2}}$ PAY PY"

_P2_REVENUE = [
    "ACCENSIONE PRESTITI",
    "ANTICIPAZIONI DA ISTITUTO TESORIERE/CASSIERE",
    "ENTRATE CORRENTI DI NATURA TRIBUTARIA, CONTRIBUTIVA E PEREQUATIVA",
    "ENTRATE DA RIDUZIONE DI ATTIVITA' FINANZIARIE",
    "ENTRATE EXTRATRIBUTARIE",
    "ENTRATE IN CONTO CAPITALE",
    "ENTRATE PER CONTO TERZI E PARTITE DI GIRO",
    "TRASFERIMENTI CORRENTI",
]
for index, category in enumerate(_P2_REVENUE, start=1):
    FEATURE_LABELS[f"Accertamenti {category}"] = rf"$E{index}_{{P2}}$ REC"
    FEATURE_LABELS[f"Riscossioni in CC {category}"] = rf"$E{index}_{{P2}}$ COL CY"
    FEATURE_LABELS[f"Riscossioni in CR {category}"] = rf"$E{index}_{{P2}}$ COL PY"


def display_feature_name(feature: str) -> str:
    """Return a catalogue code, retaining a lag suffix when one is present."""
    base = feature
    lag = None
    for separator in ("@t-", "@t"):
        if separator in feature:
            base, lag = feature.rsplit(separator, 1)
            break
    label = FEATURE_LABELS.get(base, base)
    if lag not in (None, "0"):
        label = f"{label} (t-{lag})"
    return label
