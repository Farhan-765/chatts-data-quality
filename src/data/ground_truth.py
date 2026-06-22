"""
Ground truth labels for all 55 industrial sensor signals.
Dataset: Nov 14 2025 - Feb 12 2026 (~90 days, 8640 points at 15-min sampling).
Source: Timeseer industrial plant (UHasselt research site).

Label taxonomy:
  A = Drift          (slow baseline migration over weeks)
  B = Spikes         (sudden transient outliers)
  C = Frozen         (stale / flatline)
  D = Phase Change   (abrupt permanent level shift)
  E = None/Clean     (no anomaly)
  G = Var Collapse   (intermittent amplitude collapse)
  L = Intermittent   (physically impossible values)
  B+L = Composite    (spikes + intermittent)
"""

GROUND_TRUTH = {
    # Reactor 1
    'R1-AT-101-PH':    'A',
    'R1-AT-102-COND':  'B',
    'R1-AT-103-DO':    'C',
    'R1-AT-104-TEMP':  'E',
    'R1-AT-105-PRESS': 'E',
    'R1-AT-106-ORP':   'E',

    # Reactor 2
    'R2-AT-201-NH3':   'B',
    'R2-AT-202-TOC':   'B+L',
    'R2-AT-203-PH':    'E',
    'R2-AT-204-TEMP':  'E',
    'R2-AT-205-VISC':  'E',
    'R2-AT-206-DENS':  'E',

    # Separator
    'SEP-AT-301-PH':   'A',
    'SEP-AT-302-TURB': 'B',
    'SEP-AT-303-COND': 'E',
    'SEP-AT-304-TEMP': 'E',
    'SEP-AT-305-LEVEL': 'E',
    'SEP-AT-306-DENS': 'E',

    # Distillation
    'DIST-AT-401-TEMP-TOP': 'E',
    'DIST-AT-402-TEMP-BTM': 'E',
    'DIST-AT-403-PRESS':    'E',
    'DIST-AT-404-DENS-OH':  'E',
    'DIST-AT-405-DENS-BTM': 'E',
    'DIST-AT-406-VISC':     'E',
    'DIST-AT-407-REFR':     'G',
    'DIST-AT-408-GC-C1':    'E',

    # Wastewater Treatment
    'WT-AT-501-PH':   'C',
    'WT-AT-502-COD':  'A',
    'WT-AT-503-DO':   'E',
    'WT-AT-504-TSS':  'E',
    'WT-AT-505-TURB': 'B',
    'WT-AT-506-COND': 'E',
    'WT-AT-507-NH3':  'E',
    'WT-AT-508-TOC':  'E',

    # Utilities
    'UTL-AT-601-CL2':   'C',
    'UTL-AT-602-CO2':   'B',
    'UTL-AT-603-O2':    'E',
    'UTL-AT-604-TEMP':  'E',
    'UTL-AT-605-COND':  'B',
    'UTL-AT-606-PH':    'E',
    'UTL-AT-607-TURB':  'E',
    'UTL-AT-608-FLOW':  'E',
    'UTL-AT-609-PRESS': 'E',
    'UTL-AT-610-LEVEL': 'E',

    # Packaging
    'PKG-AT-701-VISC':  'E',
    'PKG-AT-702-TEMP':  'E',
    'PKG-AT-703-VISC':  'E',
    'PKG-AT-704-TEMP':  'E',
    'PKG-AT-705-PH':    'E',
    'PKG-AT-706-COND':  'E',
    'PKG-AT-707-TURB':  'E',
    'PKG-AT-708-MOIST': 'E',
    'PKG-AT-709-COLOR': 'E',
    'PKG-AT-710-DENS':  'E',
    'PKG-AT-711-LEVEL': 'E',
}

# Tags organized by plant area (maps area name → view name → list of tags)
AREA_TAGS = {
    'Reactor 1': [
        'R1-AT-101-PH', 'R1-AT-102-COND', 'R1-AT-103-DO',
        'R1-AT-104-TEMP', 'R1-AT-105-PRESS', 'R1-AT-106-ORP',
    ],
    'Reactor 2': [
        'R2-AT-201-NH3', 'R2-AT-202-TOC', 'R2-AT-203-PH',
        'R2-AT-204-TEMP', 'R2-AT-205-VISC', 'R2-AT-206-DENS',
    ],
    'Separator': [
        'SEP-AT-301-PH', 'SEP-AT-302-TURB', 'SEP-AT-303-COND',
        'SEP-AT-304-TEMP', 'SEP-AT-305-LEVEL', 'SEP-AT-306-DENS',
    ],
    'Distillation': [
        'DIST-AT-401-TEMP-TOP', 'DIST-AT-402-TEMP-BTM', 'DIST-AT-403-PRESS',
        'DIST-AT-404-DENS-OH',  'DIST-AT-405-DENS-BTM', 'DIST-AT-406-VISC',
        'DIST-AT-407-REFR',     'DIST-AT-408-GC-C1',
    ],
    'Wastewater Treatment': [
        'WT-AT-501-PH',  'WT-AT-502-COD', 'WT-AT-503-DO',
        'WT-AT-504-TSS', 'WT-AT-505-TURB','WT-AT-506-COND',
        'WT-AT-507-NH3', 'WT-AT-508-TOC',
    ],
    'Utilities': [
        'UTL-AT-601-CL2', 'UTL-AT-602-CO2', 'UTL-AT-603-O2',
        'UTL-AT-604-TEMP','UTL-AT-605-COND','UTL-AT-606-PH',
        'UTL-AT-607-TURB','UTL-AT-608-FLOW','UTL-AT-609-PRESS',
        'UTL-AT-610-LEVEL',
    ],
    'Packaging': [
        'PKG-AT-701-VISC', 'PKG-AT-702-TEMP', 'PKG-AT-703-VISC',
        'PKG-AT-704-TEMP', 'PKG-AT-705-PH',   'PKG-AT-706-COND',
        'PKG-AT-707-TURB', 'PKG-AT-708-MOIST','PKG-AT-709-COLOR',
        'PKG-AT-710-DENS', 'PKG-AT-711-LEVEL',
    ],
}

LABEL_NAMES = {
    'A':   'Drift',
    'B':   'Spikes',
    'C':   'Frozen',
    'D':   'Phase Change',
    'E':   'None/Clean',
    'G':   'Var Collapse',
    'L':   'Intermittent Fail',
    'B+L': 'Spikes + Intermittent Fail',
    '?':   'Unclear',
}
