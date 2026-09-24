"""
Physics-Informed Thermal Demagnetization Sequence Operator (PINO-Thermal).
Models temperature-dependent spontaneous magnetization degradation Ms(T) and thermal coercivity Hc(T).
Combines:
  1. Bloch's T^{3/2} spin-wave law at low temperatures (T << Tc)
  2. 3D Heisenberg critical scaling (1 - T/Tc)^beta with beta = 0.365 near the Curie transition
  3. Paramagnetic zero-moment boundary for T >= Tc
  4. Fisher-Neel susceptibility and sublattice disorder for Antiferromagnets
"""

import numpy as np
from typing import Dict, Tuple, Optional


class PhysicsInformedThermalOperator:
    """
    Evaluates temperature-dependent magnetic order:
      - Spontaneous Polarization Ms(T)
      - Reduced Spontaneous Magnetization m(t) = Ms(T)/Ms(0) vs t = T/Tc
      - Thermal Coercivity Decay Hc(T)
      - Antiferromagnetic Sublattice Order & Fisher-Neel Susceptibility
    """
    def __init__(self):
        # 3D Heisenberg critical exponent for quantum exchange ferromagnets
        self.beta_heisenberg = 0.365
        # 3D Ising critical exponent for uniaxial highly anisotropic systems
        self.beta_ising = 0.326

    @staticmethod
    def _sanitize_inputs(tc_k: float, ms_tesla: float, hc_ka_m: float) -> Tuple[float, float, float]:
        tc = max(15.0, float(tc_k))
        ms = max(0.01, float(ms_tesla))
        hc = max(1.0, float(hc_ka_m))
        return tc, ms, hc

    def predict_thermal_profile(self, tc_k: float = 770.0, ms_tesla: float = 1.4,
                                phase: str = "FM", hc_ka_m: float = 600.0,
                                n_points: int = 160) -> Dict[str, np.ndarray]:
        """
        Evaluate full thermal degradation trajectory across T in [0, 1.25 * Tc].
        Returns dictionary containing:
          - 'temperature_k': array of temperatures
          - 'ms_t': spontaneous polarization (Tesla)
          - 'reduced_t': reduced temperature t = T/Tc
          - 'reduced_m': reduced magnetization m(t)
          - 'hc_t': thermal coercivity in kA/m
        """
        tc, ms0, hc0 = self._sanitize_inputs(tc_k, ms_tesla, hc_ka_m)
        phase_clean = str(phase).strip().upper()
        
        t_max = max(100.0, tc * 1.25)
        t_pts = np.linspace(0.0, t_max, n_points)
        reduced_t = t_pts / tc

        if phase_clean == "NM":
            # Non-magnetic: strictly zero for all temperatures
            return {
                "temperature_k": t_pts,
                "ms_t": np.zeros_like(t_pts),
                "reduced_t": reduced_t,
                "reduced_m": np.zeros_like(t_pts),
                "hc_t": np.zeros_like(t_pts),
                "label": "Non-Magnetic Ground State (Ms = 0)"
            }

        elif phase_clean == "AFM":
            # Antiferromagnet: zero net spontaneous polarization, sublattice order parameter decays up to TN
            sublattice_order = np.where(
                t_pts < tc,
                np.maximum(0.0, 1.0 - (reduced_t ** 1.6)) ** 0.34,
                0.0
            )
            ms_t = ms0 * sublattice_order
            # Fisher-Neel susceptibility: increases to peak at TN, decays as Curie-Weiss 1/(T+theta) above TN
            chi_afm = np.where(
                t_pts <= tc,
                0.35 + 0.65 * (reduced_t ** 2.0),
                1.0 / (1.0 + 1.8 * (reduced_t - 1.0))
            )
            return {
                "temperature_k": t_pts,
                "ms_t": ms_t,
                "reduced_t": reduced_t,
                "reduced_m": sublattice_order,
                "chi_afm": chi_afm,
                "hc_t": np.zeros_like(t_pts),
                "label": f"AFM Sublattice Disorder (Néel Point: {tc:.0f} K)"
            }

        else:
            # Ferromagnetic: Law of Corresponding States with Bloch spin-wave + 3D Heisenberg scaling
            # Parameter s controls low-temperature spin-wave stiffness (Bloch T^{3/2} coefficient)
            s_spinwave = 0.22  # Typical for transition metal intermetallics (Fe, Co, Nd-Fe-B)
            p_itinerant = 2.40
            
            # Universal scaled decay function
            bracket = 1.0 - s_spinwave * (reduced_t ** 1.5) - (1.0 - s_spinwave) * (reduced_t ** p_itinerant)
            bracket_safe = np.maximum(0.0, bracket)
            
            # Reduced magnetization m(t)
            reduced_m = np.where(
                t_pts < tc,
                bracket_safe ** self.beta_heisenberg,
                0.0
            )
            ms_t = ms0 * reduced_m

            # Thermal coercivity decay Hc(T) following Kronmuller micromagnetic relation
            # Hc(T) ~ Hc0 * [K1(T)/K1(0)] * [Ms(0)/Ms(T)] ~ Hc0 * (1 - T/Tc) * m(t)^1.2
            hc_t = np.where(
                t_pts < tc,
                hc0 * np.maximum(0.0, 1.0 - reduced_t) * (reduced_m ** 1.2),
                0.0
            )

            return {
                "temperature_k": t_pts,
                "ms_t": ms_t,
                "reduced_t": reduced_t,
                "reduced_m": reduced_m,
                "hc_t": hc_t,
                "label": f"Ferromagnetic Demagnetization (Curie Point: {tc:.0f} K)"
            }

    def predict_temperature_derivatives(self, tc_k: float, ms_tesla: float,
                                         temp_eval: float = 293.15) -> Dict[str, float]:
        """
        Calculate reversible temperature coefficients at Room Temperature (293 K):
          alpha(Ms) = (1/Ms) * dMs/dT (%/K)
          beta(Hc)  = (1/Hc) * dHc/dT (%/K)
        """
        tc, ms0, _ = self._sanitize_inputs(tc_k, ms_tesla, 600.0)
        if temp_eval >= tc:
            return {"alpha_ms_pct_k": 0.0, "beta_hc_pct_k": 0.0}
            
        dt = 1.0
        prof_low = self.predict_thermal_profile(tc, ms0, hc_ka_m=600.0, n_points=500)
        t_arr = prof_low["temperature_k"]
        idx = np.argmin(np.abs(t_arr - temp_eval))
        
        idx_plus = min(len(t_arr) - 1, idx + 1)
        idx_minus = max(0, idx - 1)
        
        d_temp = t_arr[idx_plus] - t_arr[idx_minus]
        d_ms = prof_low["ms_t"][idx_plus] - prof_low["ms_t"][idx_minus]
        d_hc = prof_low["hc_t"][idx_plus] - prof_low["hc_t"][idx_minus]
        
        cur_ms = max(1e-4, prof_low["ms_t"][idx])
        cur_hc = max(1e-4, prof_low["hc_t"][idx])
        
        alpha_ms = (d_ms / d_temp) / cur_ms * 100.0
        beta_hc = (d_hc / d_temp) / cur_hc * 100.0
        
        return {
            "alpha_ms_pct_k": round(float(alpha_ms), 3),
            "beta_hc_pct_k": round(float(beta_hc), 3)
        }


# Singleton operator instance
_THERMAL_OPERATOR = None

def get_thermal_operator() -> PhysicsInformedThermalOperator:
    global _THERMAL_OPERATOR
    if _THERMAL_OPERATOR is None:
        _THERMAL_OPERATOR = PhysicsInformedThermalOperator()
    return _THERMAL_OPERATOR
