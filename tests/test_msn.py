"""
Unit tests for the dedicated Magnetic Structure Network (MSN) engine.
Tests:
1. Ferromagnetic standard: BCC Fe (FM collinear, μ ~ 2.2 μB)
2. Antiferromagnetic standard: AFM Cr (Compensated sublattices, net moment ≈ 0)
3. Permanent magnet champion: Nd2Fe14B (Collinear FM, strong exchange stiffness)
4. Heavy rare-earth ferrimagnet: DyCo5 (Antiparallel Dy ⇓ vs Co ⇑ coupling)
5. Non-magnetic insulator: Al2O3 (S = 0, Diamagnetic/Grey MSG)
"""

import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymatgen.core import Structure, Lattice
from src.magnetic_structure_network import MagneticStructureNetwork, resolve_magnetic_structure



def test_msn_ferromagnetic_iron(msn):
    # BCC Fe
    lat = Lattice.cubic(2.87)
    struct = Structure(lat, ["Fe", "Fe"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    res = msn.resolve_magnetic_structure(struct, predicted_phase="FM")

    assert res["spin_alignment_flag"] == "FM"
    assert res["net_cell_moment_mu_b"] > 3.0  # 2 Fe atoms ~ 4.4 μB
    assert res["saturation_magnetization_tesla"] > 1.5  # Pure Fe is ~2.15 T
    assert res["exchange_stiffness_pj_m"] > 5.0
    assert "Type I" in res["shubnikov_msg_type"]
    assert "MAGMOM" in res["mag_cif"]
    assert len(res["site_details"]) == 2


def test_msn_antiferromagnetic_chromium(msn):
    # BCC Cr with AFM sublattices
    lat = Lattice.cubic(2.88)
    struct = Structure(lat, ["Cr", "Cr"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    res = msn.resolve_magnetic_structure(struct, predicted_phase="AFM")

    assert res["spin_alignment_flag"] == "AFM_COLLINEAR"
    assert res["net_cell_moment_mu_b"] < 0.05  # Fully compensated
    assert res["saturation_magnetization_tesla"] == 0.0
    assert res["exchange_stiffness_pj_m"] == 0.0
    assert "Type III" in res["shubnikov_msg_type"]


def test_msn_ferrimagnetic_coupling(msn):
    # DyCo5 prototype
    a, c = 4.95, 3.98
    lat = Lattice.hexagonal(a, c)
    species = ["Dy", "Co", "Co", "Co", "Co", "Co"]
    coords = [
        [0.0, 0.0, 0.0],
        [1/3, 2/3, 0.0], [2/3, 1/3, 0.0],
        [0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.5, 0.5, 0.5]
    ]
    struct = Structure(lat, species, coords)
    res = msn.resolve_magnetic_structure(struct, predicted_phase="FM")

    assert res["spin_alignment_flag"] == "FERRIMAGNETIC"
    # Dy should have negative z moment, Co positive
    dy_mom = [s["magmom_z"] for s in res["site_details"] if s["element"] == "Dy"][0]
    co_mom = [s["magmom_z"] for s in res["site_details"] if s["element"] == "Co"][0]
    assert dy_mom < 0.0  # Antiparellel
    assert co_mom > 0.0  # Parallel
    assert res["net_cell_moment_mu_b"] > 0.0  # Net uncompensated moment


def test_msn_non_magnetic_insulator(msn):
    # Al2O3 corundum
    lat = Lattice.hexagonal(4.76, 12.99)
    struct = Structure(lat, ["Al", "Al", "O", "O", "O"], [
        [0.0, 0.0, 0.35], [0.0, 0.0, 0.65],
        [0.31, 0.0, 0.25], [0.0, 0.31, 0.25], [0.69, 0.69, 0.25]
    ])
    res = msn.resolve_magnetic_structure(struct, predicted_phase="NM")

    assert res["spin_alignment_flag"] == "NON_MAGNETIC"
    assert res["net_cell_moment_mu_b"] == 0.0
    assert res["saturation_magnetization_tesla"] == 0.0
    assert res["exchange_stiffness_pj_m"] == 0.0
    assert res["num_magnetic_sites"] == 0
    assert "Type II" in res["shubnikov_msg_type"]


def test_msn_functional_api():
    lat = Lattice.cubic(3.55)
    struct = Structure(lat, ["Ni", "Ni", "Ni", "Ni"], [
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]
    ])
    res = resolve_magnetic_structure(struct, predicted_phase="FM")
    assert res["spin_alignment_flag"] == "FM"
    assert res["mean_atomic_moment_mu_b"] > 0.4


if __name__ == "__main__":
    m = MagneticStructureNetwork()
    print("Testing MSN on Fe...")
    test_msn_ferromagnetic_iron(m)
    print("Testing MSN on Cr...")
    test_msn_antiferromagnetic_chromium(m)
    print("Testing MSN on DyCo5...")
    test_msn_ferrimagnetic_coupling(m)
    print("Testing MSN on Al2O3...")
    test_msn_non_magnetic_insulator(m)
    print("Testing MSN functional API...")
    test_msn_functional_api()
    print("ALL MSN TESTS PASSED SUCCESSFULLY! (5/5)")

