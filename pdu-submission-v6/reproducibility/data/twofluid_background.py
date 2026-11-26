"""Independently integrate two-fluid hydrostatic backgrounds and compare to
the equilibrium effective-barotrope TOV construction.

Important: this validates BACKGROUND hydrostatic modelling only. There is
no relative-displacement perturbation eigenvalue calculation here.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from eos import P2G, MEV_FM3_GEO, MSUN_KM
from families import (HBARC, M0_G, ERG_CC_TO_MEV_FM3, _fermi, dm_admixed,
                      read_pp, max_mass, pc_for_mass, solve_star)
from test_tov import REF

MCHI = 938.0
G_LIST = (100.0, 300.0)
R0 = 1e-4
OUT = Path(__file__).resolve().parent / "results_twofluid_background.json"


def dark_at_k(k, g):
    k = float(max(0, k))
    n = (k/HBARC)**3/(3*np.pi**2)
    if k < 1:
        pf = k**5/(15*np.pi**2*MCHI)
        ef = n * HBARC**3 * MCHI + 1.5 * pf
    else:
        ef, pf = _fermi(k, MCHI)
    p = pf / HBARC**3 + 0.5*HBARC*g*n*n
    e = ef / HBARC**3 + 0.5*HBARC*g*n*n
    mu = np.sqrt(k*k+MCHI*MCHI) + HBARC*g*n
    return p*MEV_FM3_GEO, e*MEV_FM3_GEO, mu


def dark_from_mu(mu,g):
    if mu <= MCHI:
        return 0.0,0.0
    hi = max(5.0,np.sqrt(mu*mu-MCHI*MCHI)+1)
    k=brentq(lambda kk:dark_at_k(kk,g)[2]-mu,0,hi,xtol=1e-10)
    pp,ee,_=dark_at_k(k,g)
    return pp,ee


def integrate_separately(base,g,pc_tot):
    """Separate Euler equations dp_i/dr=-(eps_i+p_i)*nu'/2 with common metric."""
    def nuc_mu(rho):
        pn=base.p_of_rho(rho)
        en=base.eps(pn)
        return (en+pn)/P2G * ERG_CC_TO_MEV_FM3 / (rho/M0_G/1e39)

    def mix_at_rho(rho):
        pn=base.p_of_rho(rho)
        pchi,_=dark_from_mu(nuc_mu(rho),g)
        return pn+pchi,pn,pchi

    rhoc=brentq(lambda rr:mix_at_rho(rr)[0]-pc_tot,1e6,3.2e15,
                xtol=1e-5)
    _,pn0,pchi0=mix_at_rho(rhoc)
    ngrid=np.geomspace(1e-2,350,3000)
    pair=np.asarray([dark_at_k(k,g)[:2] for k in ngrid])
    pp,ee=pair[:,0],pair[:,1]
    mask=(pp>0)&(ee>0)&np.isfinite(pp)&np.isfinite(ee)
    # Tiny pressure roundoff from the relativistic Fermi integrals can
    # generate duplicate or nonmonotonic abscissae near k=0.
    order=np.argsort(pp[mask])
    pp_sort=pp[mask][order]
    ee_sort=ee[mask][order]
    p_unique,first=np.unique(pp_sort,return_index=True)
    e_unique=ee_sort[first]
    if len(p_unique)<20 or not np.all(np.diff(p_unique)>0):
        raise RuntimeError("invalid dark-fluid pressure tabulation")
    interp=PchipInterpolator(np.log(p_unique),np.log(e_unique),extrapolate=False)

    def edark(p):
        if p<=0:return 0.
        if p < p_unique[0]:
            # low-density asymptotic eps proportional to p^(3/5)
            return e_unique[0]*(p/p_unique[0])**(3/5)
        if p>p_unique[-1]:
            raise ValueError("dark pressure above interpolation interval")
        return float(np.exp(interp(np.log(p))))

    en0=base.eps(pn0)
    ec0=edark(pchi0)
    pt=pn0+pchi0
    et=en0+ec0
    factor=(2*np.pi/3)*(et+3*pt)*R0**2
    y0=np.array([pn0-(en0+pn0)*factor,
                 max(0.,pchi0-(ec0+pchi0)*factor),
                 4*np.pi/3*et*R0**3])

    def rhs(r,y):
        pn=max(y[0],base.p_surface()*1e-10)
        pc=max(y[1],0)
        m=y[2]
        en=base.eps(pn)
        ec=edark(pc)
        pt=pn+pc
        den=r*(r-2*m)
        if den<=0: raise ValueError("horizon in background integration")
        grav=(m+4*np.pi*r**3*pt)/den
        return [-(en+pn)*grav,
                -(ec+pc)*grav if y[1]>0 else 0.0,
                4*np.pi*r*r*(en+ec)]

    def event_surface(r,y):return y[0]-base.p_surface()
    event_surface.terminal=True
    event_surface.direction=-1
    sol=solve_ivp(rhs,(R0,40),y0,events=event_surface,method="DOP853",
                  rtol=2e-9,atol=[1e-13,1e-13,1e-12],max_step=0.05)
    if not sol.success or not len(sol.t_events[0]):
        raise RuntimeError(f"two-fluid background not integrated: {sol.message}")
    R=sol.t_events[0][0]
    M=sol.y_events[0][0][2]/MSUN_KM
    return dict(R_km=float(R),M_Msun=float(M),
                initial_nuclear_pressure=float(pn0),
                initial_dark_pressure=float(pchi0),
                central_nuclear_density_gcc=float(rhoc),
                dark_central_pressure_fraction=float(pchi0/pt))


def main():
    out=[]
    for name in ("SLy","AP4","MPA1"):
        a,b,c,d,*_=REF[name]
        nuclear=read_pp(a,b,c,d)
        for g in G_LIST:
            adm=dm_admixed(nuclear,g,MCHI)
            maxm,lpcm=max_mass(adm)
            central=np.exp(pc_for_mass(adm,1.4,lpcm))
            s=solve_star(adm,central,tidal=False)
            independent=integrate_separately(nuclear,g,central)
            dM=100*(independent["M_Msun"]/s["M"]-1)
            dR=100*(independent["R_km"]/s["R"]-1)
            row=dict(eos=name,G_fm2=g,Mmax_Msun=float(maxm),
                     reference_onefluid_M=float(s["M"]),reference_onefluid_R=float(s["R"]),
                     separate_background=independent,delta_M_percent=float(dM),delta_R_percent=float(dR))
            out.append(row)
            print(f"{name} G={g:.0f}: dM={dM:+.4f}% dR={dR:+.4f}%",flush=True)
    val=max(max(abs(x["delta_M_percent"]),abs(x["delta_R_percent"])) for x in out)
    result=dict(schema="two-fluid-hydrostatic-v1",explanation=__doc__,runs=out,
                summary=dict(n=len(out),max_absolute_background_discrepancy_percent=float(val)))
    OUT.write_text(json.dumps(result,indent=2))
    # Broad physical sanity criterion: report numerical discrepancies rather
    # than silently declaring the equivalence demonstrated.
    if val>5:raise AssertionError(f"two-fluid background discrepancy {val:.3f}% exceeds 5%")
    print(json.dumps(result["summary"],indent=2))


if __name__=="__main__":
    main()
