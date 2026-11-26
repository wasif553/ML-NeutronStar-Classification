"""Zhao & Lattimer, PRD 106 123002 (2022), Eqs 16,17 and Table II.
Input values transcribed from immutable catalogue 8b955ca.../population/
results_microphysics_anchor.json and pop_data/c1_0000.json through c1_0007.json.
This compares empirical fits, NOT identical-EoS eigenvalue benchmarks.
"""
import json, math
from pathlib import Path
# name, mass (solar), radius (km), frequency (Hz), damping time (s)
nuclear=[('SLy',1.4,11.705508204503252,1935.1985305204519,.1920034858781449),('AP3',1.4,11.971240728972417,1848.508910986613,.20943041064532533),('AP4',1.4,11.319226679485483,2007.6250986228606,.17927391086396605),('ENG',1.4,11.969513402325687,1845.6489569842417,.21000824562419962),('MPA1',1.4,12.43391495809553,1745.6800371086745,.23413954817654198),('WFF2',1.4,11.12450697849959,2047.4674223907625,.1729228578813333)]
admixed=[('AP3 G100',1.4,11.87306346908524,1870.665767224347,.20480063102643148),('AP3 G300',1.4,11.93751848039345,1855.8090513495654,.20795607790062515),('AP4 G100',1.4,11.227428886931124,2031.1373296708666,.1755583743279598),('AP4 G300',1.4,11.287749038195694,2015.420211948609,.17807384532934753),('ENG G100',1.4,11.868079849545865,1868.9494512907568,.20512206343141284),('ENG G300',1.4,11.934700881937353,1853.318850959668,.20845266199393225),('MPA1 G100',1.4,12.330384532176604,1767.183688511562,.2286879637498902),('MPA1 G300',1.4,12.398321973594294,1752.745951491099,.23241136594955572),('SLy G100',1.4,11.599529478661788,1960.8561926727325,.18740036977209593),('SLy G300',1.4,11.669163094034642,1943.7475100715087,.1904907963425341),('WFF2 G100',1.4,11.036774384691872,2070.888140416448,.1694569030580795),('WFF2 G300',1.4,11.094438827448686,2055.2517389371824,.17179666812921743)]
css=[('CSS 0',1.5,10.383026819028919,1927.5557963319113,.18270459356666616),('CSS 1',1.5,10.6050145931054,1850.0797155800408,.1971192061464314),('CSS 2',1.5,11.8858649739225,1577.645566344735,.26789349867793655),('CSS 3',1.5,10.203656949212013,1974.6926607833961,.174984979978273),('CSS 4',1.5,10.597104226008666,1852.1153271367593,.19671818317982934),('CSS 5',1.5,10.260416292928575,1943.2953796598617,.18019966061779455),('CSS 6',1.5,11.806189528397397,1580.879801288185,.2669345364371891),('CSS 7',1.5,10.94096049352439,1788.3851410608052,.20983067519960508)]
rows=[]
for label, data in [('nuclear',nuclear),('admixed',admixed),('self-bound',css)]:
    for name,M,R,f,tau in data:
        b=1.476625*M/R; tm=1.476625*M/299792.458
        if label=='self-bound': real=.8811*b**1.5; imag=.05352*b**4-.1849*b**5+.1253*b**6
        else: real=-.02223+.5982*b-.007331*b*b; imag=.1048*b**4-.4971*b**5+.5943*b**6
        assert real>0 and imag>0
        ff=real/(2*math.pi*tm); tt=tm/imag
        rows.append(dict(group=label,name=name,M_Msun=M,R_km=R,f_Hz=f,tau_s=tau,f_fit_Hz=ff,tau_fit_s=tt,f_residual_percent=100*(f/ff-1),tau_residual_percent=100*(tau/tt-1)))
summary={}
for label in ['nuclear','admixed','self-bound']:
    a=[r for r in rows if r['group']==label]
    summary[label]=dict(n=len(a),max_abs_f_percent=max(abs(r['f_residual_percent']) for r in a),max_abs_tau_percent=max(abs(r['tau_residual_percent']) for r in a))
out=dict(source='https://doi.org/10.1103/PhysRevD.106.123002',selection='six nuclear and twelve admixed anchor configurations at 1.4 Msun; first eight CSS catalogue indices at common tabulated mass 1.5 Msun',rows=rows,summary=summary)
Path(__file__).with_name('results_compactness_check.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(summary,indent=2))
