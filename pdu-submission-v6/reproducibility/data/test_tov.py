import numpy as np, time, sys
from scipy.optimize import brentq
from eos import *
from tov import *

REF = {  # from the f-mode inference report: logP1, G1, G2, G3, R(1.4), Lambda(1.4), f(1.4) Hz
    "SLy": (34.384, 3.005, 2.988, 2.851, 11.7061, 295.63, 1935.060),
    "AP3": (34.392, 3.166, 3.573, 3.281, 11.9717, 367.23, 1848.400),
    "AP4": (34.269, 2.830, 3.445, 3.348, 11.3197, 247.61, 2007.501),
    "ENG": (34.437, 3.514, 3.130, 3.168, 11.9699, 370.06, 1845.527),
    "MPA1": (34.495, 3.446, 3.572, 2.887, 12.4343, 479.11, 1745.580),
    "WFF2": (34.233, 2.888, 3.475, 3.517, 11.1249, 224.99, 2047.340)}


def star14(eos):
    lpc = brentq(lambda l: solve_star(eos, np.exp(l), tidal=False)["M"] - 1.4, np.log(5e-6), np.log(3e-3), xtol=1e-12)
    return solve_star(eos, np.exp(lpc))


if __name__ == "__main__":
    for gin in [float(x) for x in sys.argv[1:]] or [1.5]:
        print("inner-crust index", gin)
        for nm, (a, b, c, d, R, L, f) in REF.items():
            s = star14(read_pp(a, b, c, d, gam_in=gin))
            print("  %-5s R=%.4f (ref %.4f, %+.3f km)  Lambda=%.1f (ref %.1f, %+.1f%%)" % (nm, s["R"], R, s["R"] - R, s["Lambda"], L, 100 * (s["Lambda"] / L - 1)))
