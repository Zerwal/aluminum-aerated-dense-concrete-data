"""
Generate all figures for the paper:
"Experimental Quantification of Disproportionate Mechanical Penalties 
and Extreme Gas Retention Inefficiency in Aluminum-Powder Aerated Dense Concrete"

Run: python graphes_final.py
Figures saved as PNG in the same folder.

CONFIGURATION:
    SMOOTH_CURVES = True   → smooth spline curves (for all property plots)
    SMOOTH_CURVES = False  → straight line segments between points

    PSD plots always use straight segments (scientific convention).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline
from scipy.stats import f_oneway

# ============================================================
# *** CONFIGURATION — CHANGE THIS TO SWITCH STYLE ***
# ============================================================
SMOOTH_CURVES = True   # True = smooth spline | False = straight segments
# ============================================================

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 13, 'axes.titlesize': 14,
    'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 11, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.3,
    'lines.linewidth': 2.2, 'axes.spines.top': False, 'axes.spines.right': False,
})

Al_labels   = ['0%', '1%', '2%', '3%', '4%', '5%']
Al_num      = np.array([0, 1, 2, 3, 4, 5])
mixes_order = ['00Al','01Al','02Al','03Al','04Al','05Al']
colors_age  = {'7d':'#4e79a7','14d':'#f28e2b','28d':'#e15759'}
markers_age = {'7d':'s','14d':'^','28d':'o'}

# ============================================================
# RAW DATA
# ============================================================
comp_raw = {
    '7d':  {'00Al':[48.536,46.803,49.047],'01Al':[30.353,32.062,30.541],
            '02Al':[24.963,27.755,23.729],'03Al':[22.613,21.578,21.480],
            '04Al':[20.623,22.568,22.864],'05Al':[22.486,24.044,23.245]},
    '14d': {'00Al':[52.960,53.874,54.516],'01Al':[29.700,30.080,29.566],
            '02Al':[21.262,20.074,19.441],'03Al':[21.814,21.228,19.875],
            '04Al':[23.743,22.540,23.982],'05Al':[23.852,23.114,23.083]},
    '28d': {'00Al':[52.028,54.602,52.133],'01Al':[32.571,30.977,33.331],
            '02Al':[26.560,29.231,28.845],'03Al':[27.635,24.091,26.066],
            '04Al':[24.250,25.854,25.086],'05Al':[23.971,22.621,23.276]},
}
dens_raw = {
    '7d':  {'00Al':[2376,2353,2366],'01Al':[2332,2330,2323],'02Al':[2297,2301,2290],
            '03Al':[2264,2278,2240],'04Al':[2135,2150,2140],'05Al':[2228,2215,2210]},
    '14d': {'00Al':[2465,2493,2450],'01Al':[2266,2278,2258],'02Al':[2235,2210,2216],
            '03Al':[2193,2182,2183],'04Al':[2273,2248,2271],'05Al':[2279,2247,2233]},
    '28d': {'00Al':[2396,2369,2388],'01Al':[2373,2341,2335],'02Al':[2249,2243,2267],
            '03Al':[2182,2174,2189],'04Al':[2198,2190,2184],'05Al':[2199,2164,2189]},
}
porosity_raw = {
    '00Al':[3.20,3.29,3.22],'01Al':[4.72,4.83,5.07],
    '02Al':[5.29,5.21,5.12],'03Al':[5.89,5.92,5.68],
    '04Al':[5.34,5.54,5.52],'05Al':[6.19,6.16,6.18],
}
split_raw = {
    '00Al':[3.554,3.451,3.577],'01Al':[3.448,3.535,3.412],
    '02Al':[3.820,3.335,3.132],'03Al':[3.425,3.446,3.421],
    '04Al':[2.427,2.447,2.584],'05Al':[2.469,2.316,2.452],
}
flex_mean_arr = np.array([9.36,8.09,7.68,6.65,6.18,5.89])
flex_sd_arr   = np.array([0.25,0.30,0.28,0.35,0.25,0.20])

time_h = [0, 10/60, 20/60, 1, 2, 5, 24, 48, 72, 192, 216, 240]
mass_t = {
    '00Al':[2365.67,2365.00,2364.67,2363.00,2360.33,2356.00,2341.33,2338.00,2334.33,2324.67,2324.00,2324.00],
    '01Al':[2275.33,2274.67,2274.00,2272.00,2269.33,2264.67,2247.67,2243.67,2239.33,2227.33,2226.00,2226.00],
    '02Al':[2237.00,2236.67,2236.33,2234.00,2231.33,2226.33,2209.67,2204.67,2200.00,2187.67,2185.67,2185.67],
    '03Al':[2155.00,2154.33,2153.33,2150.67,2147.67,2142.00,2125.00,2119.33,2114.00,2101.33,2099.33,2099.33],
    '04Al':[2169.00,2168.67,2168.00,2165.33,2162.33,2157.00,2141.33,2136.00,2131.00,2118.00,2116.67,2116.67],
    '05Al':[2192.50,2191.50,2191.00,2188.50,2183.50,2175.50,2153.00,2147.00,2141.00,2126.50,2125.00,2125.00],
}
M0 = {k:v[0] for k,v in mass_t.items()}
massloss = {mix:[(M0[mix]-mt)/M0[mix]*100 for mt in mass_t[mix]] for mix in mixes_order}

fa_sieve_mass = [831,808,785,758,709,711,580,712]
fa_sieve_agg  = {
    'S1':[849,886,1353,1573,1132,776,607,714],
    'S2':[852,886,1143,1310,1443,877,665,716],
    'S3':[860,912,1220,1350,1315,864,653,718],
}
fa_sieves_mm = [5.0,2.36,1.25,0.60,0.30,0.18,0.075,0.0]
ca_sieve_mass = [985,1027,1084,990,815,831,808,712]
ca_sieve_agg  = {
    'S1':[985,1027,1115,1152,1614,2801,845,713],
    'S2':[985,1027,1101,1146,1520,2848,911,714],
    'S3':[985,1027,1100,1164,1509,2910,844,713],
}
ca_sieves_mm = [20.0,16.0,14.0,12.5,10.0,5.0,2.36,0.0]
astm_fa_sieves=[9.5,4.75,2.36,1.18,0.60,0.30,0.15]
astm_fa_upper =[100,100,100,85,60,30,10]
astm_fa_lower =[100,95,80,50,25,5,0]
astm_ca_sieves=[20.0,14.0,10.0,5.0,2.36]
astm_ca_upper =[100,100,85,20,5]
astm_ca_lower =[100,90,40,0,0]

# ============================================================
# HELPERS
# ============================================================
def mean_sd(d, keys=mixes_order):
    m,s=[],[]
    for k in keys:
        v=d[k]; m.append(np.mean(v)); s.append(np.std(v,ddof=1))
    return np.array(m),np.array(s)

def get_line(x, y, n=300):
    """Returns smooth spline or straight segments depending on SMOOTH_CURVES."""
    x,y = np.array(x,dtype=float), np.array(y,dtype=float)
    idx = np.argsort(x); x,y = x[idx],y[idx]
    if SMOOTH_CURVES:
        xs = np.linspace(x.min(), x.max(), n)
        k  = min(3, len(x)-1)
        return xs, make_interp_spline(x, y, k=k)(xs)
    else:
        return x, y   # straight segments

def rd_model(P,f0,b): return f0*np.exp(-b*P)
def pw_model(P,a,n):  return a*P**n
def rmse(yt,yp): return np.sqrt(np.mean((yt-yp)**2))
def r2(yt,yp):   return 1-np.sum((yt-yp)**2)/np.sum((yt-np.mean(yt))**2)

def psd_passing(sieve_masses, sieve_agg_dict):
    results={}
    for name,agg in sieve_agg_dict.items():
        net=[a-s for a,s in zip(agg,sieve_masses)]
        cumul=[]; c=0
        for n in net: c+=n; cumul.append(c)
        total=cumul[-1]
        results[name]=[100-(c/total*100) for c in cumul]
    return results

# ============================================================
# DERIVED STATISTICS
# ============================================================
comp_mean,comp_sd={},{}
for age in ['7d','14d','28d']:
    comp_mean[age],comp_sd[age]=mean_sd(comp_raw[age])
dens_mean,dens_sd={},{}
for age in ['7d','14d','28d']:
    dens_mean[age],dens_sd[age]=mean_sd(dens_raw[age])
por_mean,por_sd     = mean_sd(porosity_raw)
split_mean,split_sd = mean_sd(split_raw)
ft_fc        = split_mean/comp_mean['28d']
foaming_eff  = np.diff(por_mean)

Al_kg  = np.array([0,5.23,10.461,15.691,20.921,26.152])
n_H2   = 1.5*(Al_kg*1000/26.98)
V_H2   = n_H2*24.5/1000
V_pore = por_mean/100
eta    = np.where(V_H2>0, V_pore/V_H2*100, 0)

fc28=comp_mean['28d']; rho28=dens_mean['28d']
fc_loss =[(fc28[0]-f)/fc28[0]*100 for f in fc28]
rho_loss=[(rho28[0]-r)/rho28[0]*100 for r in rho28]

F_comp,_  = f_oneway(*[comp_raw['28d'][m] for m in mixes_order])
F_split,_ = f_oneway(*[split_raw[m] for m in mixes_order])

fa_passing   = psd_passing(fa_sieve_mass, fa_sieve_agg)
ca_passing   = psd_passing(ca_sieve_mass, ca_sieve_agg)
fa_mean_pass = np.array([np.mean([fa_passing[s][i] for s in ['S1','S2','S3']]) for i in range(len(fa_sieves_mm))])
fa_std_pass  = np.array([np.std( [fa_passing[s][i] for s in ['S1','S2','S3']],ddof=1) for i in range(len(fa_sieves_mm))])
ca_mean_pass = np.array([np.mean([ca_passing[s][i] for s in ['S1','S2','S3']]) for i in range(len(ca_sieves_mm))])
ca_std_pass  = np.array([np.std( [ca_passing[s][i] for s in ['S1','S2','S3']],ddof=1) for i in range(len(ca_sieves_mm))])
fm_values = []
for name in ['S1','S2','S3']:
    pct_ret=[100-p for p in fa_passing[name]]
    fm_values.append(sum(pct_ret[:6])/100)
fm_mean=np.mean(fm_values)

suffix = "_smooth" if SMOOTH_CURVES else "_segments"
print(f"Mode: {'SMOOTH CURVES' if SMOOTH_CURVES else 'STRAIGHT SEGMENTS'}")
print(f"Files will be saved with suffix: {suffix}\n")

# ============================================================
# FIG 1 — Porosity & Density
# ============================================================
fig,axes=plt.subplots(1,2,figsize=(13,5))
ax1=axes[0]
xs,ys=get_line(Al_num,por_mean)
ax1.plot(xs,ys,'-',color='#0f3460',linewidth=2.5,zorder=3)
ax1.errorbar(Al_num,por_mean,yerr=por_sd,fmt='o',color='#0f3460',
             capsize=5,capthick=1.8,elinewidth=1.8,markersize=8,zorder=5,label='Apparent porosity')
ax1.set_xlabel('Aluminum particle content (% by mass of cement)')
ax1.set_ylabel('Apparent porosity (%)')
ax1.set_title('(a) Apparent porosity vs aluminum particle content')
ax1.set_xticks(Al_num); ax1.set_xticklabels(Al_labels)
ax1.set_ylim(0,8); ax1.legend()

ax2=axes[1]
for age in ['7d','14d','28d']:
    xs2,ys2=get_line(Al_num,dens_mean[age]/1000)
    ax2.plot(xs2,ys2,'-',color=colors_age[age],linewidth=2.2,zorder=3)
    ax2.errorbar(Al_num,dens_mean[age]/1000,yerr=dens_sd[age]/1000,
                 fmt=markers_age[age],color=colors_age[age],
                 capsize=4,capthick=1.5,elinewidth=1.5,markersize=7,zorder=5,label=age)
ax2.set_xlabel('Aluminum particle content (% by mass of cement)')
ax2.set_ylabel('Dry density (g/cm³)')
ax2.set_title('(b) Dry density vs aluminum particle content')
ax2.set_xticks(Al_num); ax2.set_xticklabels(Al_labels)
ax2.set_ylim(1.5,3.0); ax2.legend()
plt.tight_layout(); plt.savefig(f'Fig1_porosity_density{suffix}.png'); plt.close()
print(f"Saved: Fig1_porosity_density{suffix}.png")

# ============================================================
# FIG 2 — Compressive strength all ages
# ============================================================
fig,ax=plt.subplots(figsize=(9,5.5))
for age in ['7d','14d','28d']:
    xs,ys=get_line(Al_num,comp_mean[age])
    ax.plot(xs,ys,'-',color=colors_age[age],linewidth=2.5,zorder=3)
    ax.errorbar(Al_num,comp_mean[age],yerr=comp_sd[age],
                fmt=markers_age[age],color=colors_age[age],
                capsize=5,capthick=1.8,elinewidth=1.8,markersize=8,zorder=5,label=age)
ax.set_xlabel('Aluminum particle content (% by mass of cement)')
ax.set_ylabel('Compressive strength (MPa)')
ax.set_title('Compressive strength development with aluminum particle content')
ax.set_xticks(Al_num); ax.set_xticklabels(Al_labels)
ax.set_ylim(0,65); ax.legend()
plt.tight_layout(); plt.savefig(f'Fig2_compressive_strength{suffix}.png'); plt.close()
print(f"Saved: Fig2_compressive_strength{suffix}.png")

# ============================================================
# FIG 3 — Three Regimes
# ============================================================
fig,ax=plt.subplots(figsize=(9,5.5))
xs,ys=get_line(Al_num,comp_mean['28d'])
ax.plot(xs,ys,'-',color='black',linewidth=2.5,zorder=4)
ax.errorbar(Al_num,comp_mean['28d'],yerr=comp_sd['28d'],fmt='o',color='black',
            capsize=5,capthick=1.8,elinewidth=1.8,markersize=9,zorder=5)
ax.axvspan(-0.2,1.2,alpha=0.13,color='#e15759',label='Regime 1: Sharp drop (0–1% Al)')
ax.axvspan(1.2,3.2,alpha=0.13,color='#f28e2b',label='Regime 2: Progressive weakening (1–3% Al)')
ax.axvspan(3.2,5.2,alpha=0.13,color='#59a14f',label='Regime 3: Near-saturation (3–5% Al)')
ax.text(0.5,44,'Regime 1\n−39%',ha='center',color='#c0392b',fontsize=12,fontweight='bold')
ax.text(2.0,31,'Regime 2\n−19.7%',ha='center',color='#d97000',fontsize=12,fontweight='bold')
ax.text(4.0,28,'Regime 3\n−10.2%',ha='center',color='#2a7a00',fontsize=12,fontweight='bold')
ax.set_xlabel('Aluminum particle content (% by mass of cement)')
ax.set_ylabel('28-day compressive strength (MPa)')
ax.set_title('Three behavioral regimes of compressive strength degradation')
ax.set_xticks(Al_num); ax.set_xticklabels(Al_labels)
ax.set_xlim(-0.3,5.3); ax.set_ylim(0,65)
ax.legend(loc='upper right',framealpha=0.9)
plt.tight_layout(); plt.savefig(f'Fig3_three_regimes{suffix}.png'); plt.close()
print(f"Saved: Fig3_three_regimes{suffix}.png")

# ============================================================
# FIG 4 — Strength-Porosity R-D + Power-law + RMSE
# ============================================================
fig,axes=plt.subplots(1,3,figsize=(16,5.5))
for idx,(age,ax) in enumerate(zip(['7d','14d','28d'],axes)):
    fc=comp_mean[age]; P=por_mean
    popt_rd,_=curve_fit(rd_model,P,fc,p0=[100,0.3])
    popt_pw,_=curve_fit(pw_model,P,fc,p0=[200,-1])
    fc_rd=rd_model(P,*popt_rd); fc_pw=pw_model(P,*popt_pw)
    R2_rd=r2(fc,fc_rd); RMSE_rd=rmse(fc,fc_rd)
    R2_pw=r2(fc,fc_pw); RMSE_pw=rmse(fc,fc_pw)
    por_fit=np.linspace(2.8,6.5,200)  # regression lines always smooth
    ax.errorbar(P,fc,yerr=comp_sd[age],fmt='o',color='#0f3460',
                capsize=4,capthick=1.5,markersize=8,zorder=5,label='Experimental')
    ax.plot(por_fit,rd_model(por_fit,*popt_rd),'-',color='#e15759',linewidth=2.2,
            label=f'R–D: f₀={popt_rd[0]:.1f}, b={popt_rd[1]:.3f}\nR²={R2_rd:.3f}, RMSE={RMSE_rd:.2f} MPa')
    ax.plot(por_fit,pw_model(por_fit,*popt_pw),'--',color='#59a14f',linewidth=2.2,
            label=f'Power: a={popt_pw[0]:.1f}, n={popt_pw[1]:.3f}\nR²={R2_pw:.3f}, RMSE={RMSE_pw:.2f} MPa')
    ax.set_xlabel('Apparent porosity (%)')
    ax.set_ylabel('Compressive strength (MPa)')
    ax.set_title(f'({chr(97+idx)}) {age} — Strength–porosity')
    ax.legend(fontsize=9,loc='upper right'); ax.set_ylim(0,65)
plt.tight_layout(); plt.savefig(f'Fig4_porosity_strength_regression{suffix}.png'); plt.close()
print(f"Saved: Fig4_porosity_strength_regression{suffix}.png")

# ============================================================
# FIG 5 — Split + Flexural
# ============================================================
fig,axes=plt.subplots(1,2,figsize=(13,5))
ax1=axes[0]
xs,ys=get_line(Al_num,split_mean)
ax1.plot(xs,ys,'-',color='#533483',linewidth=2.5,zorder=3)
ax1.errorbar(Al_num,split_mean,yerr=split_sd,fmt='s',color='#533483',
             capsize=5,capthick=1.8,elinewidth=1.8,markersize=8,zorder=5)
ax1.set_xlabel('Aluminum particle content (% by mass of cement)')
ax1.set_ylabel('Splitting tensile strength (MPa)')
ax1.set_title(f'(a) Split tensile strength (28d)\nANOVA: F={F_split:.1f}, p<0.001')
ax1.set_xticks(Al_num); ax1.set_xticklabels(Al_labels); ax1.set_ylim(0,5)

ax2=axes[1]
xs2,ys2=get_line(Al_num,flex_mean_arr)
ax2.plot(xs2,ys2,'-',color='#e94560',linewidth=2.5,zorder=3)
ax2.errorbar(Al_num,flex_mean_arr,yerr=flex_sd_arr,fmt='^',color='#e94560',
             capsize=5,capthick=1.8,elinewidth=1.8,markersize=8,zorder=5)
ax2.set_xlabel('Aluminum particle content (% by mass of cement)')
ax2.set_ylabel('Flexural strength / MOR (MPa)')
ax2.set_title('(b) Flexural strength (MOR, 28d)')
ax2.set_xticks(Al_num); ax2.set_xticklabels(Al_labels); ax2.set_ylim(0,12)
plt.tight_layout(); plt.savefig(f'Fig5_split_flexural{suffix}.png'); plt.close()
print(f"Saved: Fig5_split_flexural{suffix}.png")

# ============================================================
# FIG 6 — ft/fc ratio
# ============================================================
fig,ax=plt.subplots(figsize=(8,5))
xs,ys=get_line(Al_num,ft_fc)
ax.plot(xs,ys,'-',color='#0f3460',linewidth=2.5,zorder=3)
ax.plot(Al_num,ft_fc,'D',color='#0f3460',markersize=9,zorder=5)
for x,y in zip(Al_num,ft_fc):
    ax.annotate(f'{y:.3f}',(x,y),textcoords='offset points',xytext=(0,12),
                ha='center',fontsize=11,fontweight='bold')
ax.set_xlabel('Aluminum particle content (% by mass of cement)')
ax.set_ylabel('ft / fc ratio (–)')
ax.set_title('Evolution of the tensile-to-compressive strength ratio (ft/fc)')
ax.set_xticks(Al_num); ax.set_xticklabels(Al_labels); ax.set_ylim(0,0.20)
plt.tight_layout(); plt.savefig(f'Fig6_ft_fc_ratio{suffix}.png'); plt.close()
print(f"Saved: Fig6_ft_fc_ratio{suffix}.png")

# ============================================================
# FIG 7 — Gas retention efficiency
# ============================================================
fig,axes=plt.subplots(1,2,figsize=(14,5))
ax1=axes[0]
x_eff=[0.5,1.5,2.5,3.5,4.5]
colors_bar=['#59a14f' if v>0 else '#e15759' for v in foaming_eff]
bars=ax1.bar(x_eff,foaming_eff,width=0.6,color=colors_bar,alpha=0.8,edgecolor='black',linewidth=0.8)
ax1.axhline(0,color='black',linewidth=0.8)
ax1.set_xlabel('Aluminum particle content interval')
ax1.set_ylabel('Incremental porosity increase (% per 1% Al)')
ax1.set_title('(a) Incremental gas retention efficiency')
ax1.set_xticks(x_eff); ax1.set_xticklabels(['0→1%','1→2%','2→3%','3→4%','4→5%'])
for bar,v in zip(bars,foaming_eff):
    ypos=bar.get_height()+0.01 if v>=0 else bar.get_height()-0.06
    ax1.text(bar.get_x()+bar.get_width()/2,ypos,f'{v:.3f}%',ha='center',va='bottom',fontsize=11,fontweight='bold')

ax2=axes[1]
eta_plot=eta[1:]; al_plot=Al_num[1:]
xs3,ys3=get_line(al_plot,eta_plot)
ax2.plot(xs3,ys3,'-',color='#e15759',linewidth=2.5,zorder=3)
ax2.plot(al_plot,eta_plot,'o',color='#e15759',markersize=9,zorder=5)
ax2.fill_between(xs3,ys3,alpha=0.15,color='#e15759')
for x,y in zip(al_plot,eta_plot):
    ax2.annotate(f'{y:.3f}%',(x,y),textcoords='offset points',xytext=(0,10),
                ha='center',fontsize=11,fontweight='bold')
ax2.set_xlabel('Aluminum particle content (% by mass of cement)')
ax2.set_ylabel('Gas retention efficiency η (%)')
ax2.set_title('(b) Stoichiometric gas retention efficiency\n(V_pore / V_H₂_theoretical)')
ax2.set_xticks(al_plot); ax2.set_xticklabels(['1%','2%','3%','4%','5%']); ax2.set_ylim(0,1.0)
plt.tight_layout(); plt.savefig(f'Fig7_gas_retention_efficiency{suffix}.png'); plt.close()
print(f"Saved: Fig7_gas_retention_efficiency{suffix}.png")

# ============================================================
# FIG 8 — Disproportionate strength vs density loss
# ============================================================
fig,ax=plt.subplots(figsize=(9,5.5))
ax2t=ax.twinx()
xs_fc,ys_fc=get_line(Al_num,fc_loss)
xs_rh,ys_rh=get_line(Al_num,rho_loss)
ax.plot(xs_fc,ys_fc,'-',color='#e15759',linewidth=2.5,zorder=3)
ax.plot(Al_num,fc_loss,'o',color='#e15759',markersize=9,zorder=5,label='Strength loss (%)')
ax2t.plot(xs_rh,ys_rh,'-',color='#4e79a7',linewidth=2.5,zorder=3)
ax2t.plot(Al_num,rho_loss,'s',color='#4e79a7',markersize=9,zorder=5,label='Density loss (%)')
ax.set_xlabel('Aluminum particle content (% by mass of cement)')
ax.set_ylabel('Compressive strength loss (%)',color='#e15759',fontsize=13)
ax2t.set_ylabel('Density loss (%)',color='#4e79a7',fontsize=13)
ax.set_xticks(Al_num); ax.set_xticklabels(Al_labels)
ax.set_title('Disproportionate compressive strength loss vs density reduction')
ax.set_ylim(0,70); ax2t.set_ylim(0,15)
ax.tick_params(axis='y',labelcolor='#e15759')
ax2t.tick_params(axis='y',labelcolor='#4e79a7')
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2t.get_legend_handles_labels()
ax.legend(h1+h2,l1+l2,loc='upper left',framealpha=0.9)
plt.tight_layout(); plt.savefig(f'Fig8_strength_vs_density_loss{suffix}.png'); plt.close()
print(f"Saved: Fig8_strength_vs_density_loss{suffix}.png")

# ============================================================
# FIG 9 — Normalized strengths (bar chart — no line style switch needed)
# ============================================================
fig,ax=plt.subplots(figsize=(11,5.5))
x=np.arange(len(Al_labels)); w=0.25
fc_norm=comp_mean['28d']/comp_mean['28d'][0]*100
ft_norm=split_mean/split_mean[0]*100
fr_norm=flex_mean_arr/flex_mean_arr[0]*100
ax.bar(x-w,fc_norm,w,label='Compressive fc',color='#0f3460',alpha=0.85,edgecolor='black',linewidth=0.5)
ax.bar(x,ft_norm,w,label='Split tensile ft',color='#533483',alpha=0.85,edgecolor='black',linewidth=0.5)
ax.bar(x+w,fr_norm,w,label='Flexural fr (MOR)',color='#e94560',alpha=0.85,edgecolor='black',linewidth=0.5)
ax.axhline(100,color='gray',linestyle='--',linewidth=1.2,alpha=0.7)
ax.set_xlabel('Aluminum particle content (% by mass of cement)')
ax.set_ylabel('Relative strength (% of control)')
ax.set_title('Normalized 28-day mechanical strengths relative to control mix')
ax.set_xticks(x); ax.set_xticklabels(Al_labels)
ax.legend(framealpha=0.9); ax.set_ylim(0,125)
plt.tight_layout(); plt.savefig(f'Fig9_normalized_strengths{suffix}.png'); plt.close()
print(f"Saved: Fig9_normalized_strengths{suffix}.png")

# ============================================================
# FIG 10 — Drying Kinetics
# ============================================================
drying_colors=['#1a1a2e','#16213e','#0f3460','#533483','#e94560','#f5a623']
fig,ax=plt.subplots(figsize=(10,5.5))
for mix,col in zip(mixes_order,drying_colors):
    ax.plot(time_h,massloss[mix],'o',color=col,markersize=5,zorder=5)
    xs_d,ys_d=get_line(time_h,massloss[mix])
    ax.plot(xs_d,ys_d,'-',color=col,linewidth=2.2,label=f'{mix.replace("Al","% Al")}')
ax.set_xlabel('Time (hours)')
ax.set_ylabel('Mass loss (%)')
ax.set_title('Drying kinetics under ambient conditions (25±2°C, 60±5% RH)')
ax.legend(loc='lower right',framealpha=0.9); ax.set_xlim(-2,250)
plt.tight_layout(); plt.savefig(f'Fig10_drying_kinetics{suffix}.png'); plt.close()
print(f"Saved: Fig10_drying_kinetics{suffix}.png")

# ============================================================
# FIG 11 — PSD (ALWAYS straight segments — scientific convention)
# ============================================================
fig,axes=plt.subplots(1,2,figsize=(14,5.5))
ax1=axes[0]
ax1.fill_between(astm_fa_sieves,astm_fa_lower,astm_fa_upper,
                 alpha=0.18,color='#f28e2b',label='ASTM C33 limits')
ax1.plot(astm_fa_sieves,astm_fa_upper,'--',color='#f28e2b',linewidth=1.2,alpha=0.8)
ax1.plot(astm_fa_sieves,astm_fa_lower,'--',color='#f28e2b',linewidth=1.2,alpha=0.8)
xp_fa=np.array(fa_sieves_mm[:-1])
ax1.plot(xp_fa,fa_mean_pass[:-1],'s-',color='#0f3460',linewidth=2.5,markersize=8,zorder=5,label='Mean')
ax1.fill_between(xp_fa,
                 fa_mean_pass[:-1]-fa_std_pass[:-1],
                 fa_mean_pass[:-1]+fa_std_pass[:-1],
                 alpha=0.18,color='#0f3460',label='Mean ± SD')
ax1.set_xscale('log'); ax1.set_xlim(0.05,12); ax1.set_ylim(-2,105)
ax1.set_xlabel('Sieve size (mm)'); ax1.set_ylabel('Cumulative percentage passing (%)')
ax1.set_title('(a) Fine aggregate — Particle size distribution')
ax1.set_xticks([0.075,0.15,0.30,0.60,1.18,2.36,4.75])
ax1.set_xticklabels(['0.075','0.15','0.30','0.60','1.18','2.36','4.75'])
ax1.legend(loc='upper left',fontsize=11,framealpha=0.9)
ax1.text(0.08,8,f'FM = {fm_mean:.2f}',fontsize=12,
         bbox=dict(boxstyle='round,pad=0.3',facecolor='white',edgecolor='#0f3460',alpha=0.9))

ax2=axes[1]
ax2.fill_between(astm_ca_sieves,astm_ca_lower,astm_ca_upper,
                 alpha=0.18,color='#e15759',label='ASTM C33 limits')
ax2.plot(astm_ca_sieves,astm_ca_upper,'--',color='#e15759',linewidth=1.2,alpha=0.8)
ax2.plot(astm_ca_sieves,astm_ca_lower,'--',color='#e15759',linewidth=1.2,alpha=0.8)
xp_ca=np.array(ca_sieves_mm[:-1])
ax2.plot(xp_ca,ca_mean_pass[:-1],'s-',color='#0f3460',linewidth=2.5,markersize=8,zorder=5,label='Mean')
ax2.fill_between(xp_ca,
                 ca_mean_pass[:-1]-ca_std_pass[:-1],
                 ca_mean_pass[:-1]+ca_std_pass[:-1],
                 alpha=0.18,color='#0f3460',label='Mean ± SD')
ax2.set_xscale('log'); ax2.set_xlim(1.5,25); ax2.set_ylim(-2,105)
ax2.set_xlabel('Sieve size (mm)'); ax2.set_ylabel('Cumulative percentage passing (%)')
ax2.set_title('(b) Coarse aggregate — Particle size distribution')
ax2.set_xticks([2.36,5.0,10.0,12.5,14.0,16.0,20.0])
ax2.set_xticklabels(['2.36','5','10','12.5','14','16','20'])
ax2.legend(loc='upper left',fontsize=11,framealpha=0.9)
ax2.axvline(x=14,color='#0f3460',linestyle=':',linewidth=1.5,alpha=0.7)
ax2.text(14.3,50,'D$_{max}$ = 14 mm',fontsize=11,color='#0f3460',rotation=90,va='center')
# Note: PSD always uses straight segments regardless of SMOOTH_CURVES setting
plt.tight_layout(); plt.savefig('Fig11_PSD.png'); plt.close()
print("Saved: Fig11_PSD.png  (always straight segments — scientific convention)")

print(f"\n{'='*55}")
print(f"All 11 figures generated in {'SMOOTH' if SMOOTH_CURVES else 'SEGMENTS'} mode!")
print(f"{'='*55}")

# ============================================================
# UNIFIED MODEL SECTION — append to graphes_final.py
# ============================================================
from mpl_toolkits.mplot3d import Axes3D

# ── Data for unified model ──
ages_days = {'7d':7,'14d':14,'28d':28}
P_all,t_all,fc_all=[],[],[]
for age,t in ages_days.items():
    for p,f in zip(por_mean,comp_mean[age]):
        P_all.append(p); t_all.append(t); fc_all.append(f)
P_all=np.array(P_all); t_all=np.array(t_all,dtype=float); fc_all=np.array(fc_all)

def unified_model(X,A,alpha,beta):
    P,t=X; return A*(t**alpha)*(P**beta)

popt_uni,pcov_uni=curve_fit(unified_model,(P_all,t_all),fc_all,
                            p0=[500,0.2,-2.0],maxfev=10000)
A_opt,alpha_opt,beta_opt=popt_uni
perr_uni=np.sqrt(np.diag(pcov_uni))
fc_pred_all=unified_model((P_all,t_all),*popt_uni)
R2_uni=r2(fc_pred_all,fc_all)
RMSE_uni=rmse(fc_all,fc_pred_all)

col_map={'7d':'#e15759','14d':'#f28e2b','28d':'#4e79a7'}

# ── FIG: Model comparison ──
fig,axes=plt.subplots(1,3,figsize=(16,5.5))
por_fit=np.linspace(2.8,6.5,200)
for idx,(age,t) in enumerate(ages_days.items()):
    ax=axes[idx]; fc=comp_mean[age]; P=por_mean
    popt_rd,_=curve_fit(rd_model,P,fc,p0=[100,0.3])
    popt_pw,_=curve_fit(pw_model,P,fc,p0=[200,-1])
    yp_rd=rd_model(P,*popt_rd); yp_pw=pw_model(P,*popt_pw)
    yp_un=unified_model((P,np.full(6,float(t))),*popt_uni)
    ax.errorbar(P,fc,yerr=comp_sd[age],fmt=markers_age[age],color='black',
                capsize=4,capthick=1.5,markersize=9,zorder=6,label='Experimental')
    ax.plot(por_fit,rd_model(por_fit,*popt_rd),'-',color='#e15759',linewidth=2.0,
            label=f'R–D  R²={r2(fc,yp_rd):.3f}, RMSE={rmse(fc,yp_rd):.2f}')
    ax.plot(por_fit,pw_model(por_fit,*popt_pw),'--',color='#59a14f',linewidth=2.0,
            label=f'Power  R²={r2(fc,yp_pw):.3f}, RMSE={rmse(fc,yp_pw):.2f}')
    ax.plot(por_fit,unified_model((por_fit,np.full(len(por_fit),float(t))),*popt_uni),
            ':',color='#533483',linewidth=2.5,
            label=f'Unified  R²={r2(fc,yp_un):.3f}, RMSE={rmse(fc,yp_un):.2f}')
    ax.set_xlabel('Apparent porosity (%)'); ax.set_ylabel('Compressive strength (MPa)')
    ax.set_title(f'({chr(97+idx)}) {age} — Model comparison')
    ax.legend(fontsize=9,loc='upper right'); ax.set_ylim(0,65)
plt.suptitle('Comparison of R–D, Power-law, and Unified models across curing ages',
             fontsize=13,fontweight='bold',y=1.02)
plt.tight_layout()
plt.savefig(f'Fig_model_comparison{suffix}.png'); plt.close()
print(f"Saved: Fig_model_comparison{suffix}.png")

# ── FIG: 3D surface ──
fig=plt.figure(figsize=(11,7))
ax=fig.add_subplot(111,projection='3d')
P_grid=np.linspace(2.5,7.0,50); t_grid=np.linspace(5,32,50)
PP,TT=np.meshgrid(P_grid,t_grid)
FC=unified_model((PP.ravel(),TT.ravel()),*popt_uni).reshape(PP.shape)
surf=ax.plot_surface(PP,TT,FC,cmap='viridis',alpha=0.75,edgecolor='none')
for age,t in ages_days.items():
    ax.scatter(por_mean,np.full(6,float(t)),comp_mean[age],
               color=col_map[age],s=80,zorder=5,label=f'Exp. {age}',depthshade=False)
fig.colorbar(surf,ax=ax,shrink=0.5,label='fc (MPa)')
ax.set_xlabel('Porosity (%)',labelpad=8); ax.set_ylabel('Curing age (days)',labelpad=8)
ax.set_zlabel('fc (MPa)',labelpad=8)
ax.set_title(f'Unified model: fc = {A_opt:.1f} · t^{alpha_opt:.3f} · P^{beta_opt:.3f}\n'
             f'R² = {R2_uni:.4f},  RMSE = {RMSE_uni:.2f} MPa  (n=18)',fontsize=12)
ax.legend(loc='upper right',fontsize=10); ax.view_init(elev=25,azim=-50)
plt.tight_layout()
plt.savefig('Fig_unified_model_3D.png'); plt.close()
print("Saved: Fig_unified_model_3D.png")

# ── FIG: Residuals ──
residuals=fc_all-fc_pred_all
colors_pts=[col_map[age] for age in ['7d']*6+['14d']*6+['28d']*6]
fig,axes=plt.subplots(1,2,figsize=(13,5))
ax1=axes[0]
ax1.scatter(fc_pred_all,residuals,c=colors_pts,s=80,zorder=5,edgecolors='black',linewidth=0.5)
ax1.axhline(0,color='black',linewidth=1.5,linestyle='--')
ax1.axhline(2,color='gray',linewidth=1,linestyle=':',alpha=0.6)
ax1.axhline(-2,color='gray',linewidth=1,linestyle=':',alpha=0.6)
ax1.set_xlabel('Predicted fc (MPa)'); ax1.set_ylabel('Residual (MPa)')
ax1.set_title('Residual plot — Unified model')
from matplotlib.patches import Patch
ax1.legend(handles=[Patch(color=col_map[a],label=a) for a in ['7d','14d','28d']],fontsize=11)
ax2=axes[1]
ax2.scatter(fc_all,fc_pred_all,c=colors_pts,s=80,zorder=5,edgecolors='black',linewidth=0.5)
lims=[15,60]; ax2.plot(lims,lims,'k--',linewidth=1.5,label='1:1 line')
ax2.set_xlabel('Observed fc (MPa)'); ax2.set_ylabel('Predicted fc (MPa)')
ax2.set_title('Observed vs Predicted — Unified model')
ax2.set_xlim(lims); ax2.set_ylim(lims)
ax2.legend(handles=[Patch(color=col_map[a],label=a) for a in ['7d','14d','28d']]+
           [plt.Line2D([0],[0],color='k',linestyle='--',label='1:1 line')],fontsize=11)
plt.tight_layout()
plt.savefig(f'Fig_unified_model_residuals{suffix}.png'); plt.close()
print(f"Saved: Fig_unified_model_residuals{suffix}.png")

print(f"\nUnified model: fc = {A_opt:.2f} · t^{alpha_opt:.4f} · P^{beta_opt:.4f}")
print(f"R² = {R2_uni:.4f}, RMSE = {RMSE_uni:.4f} MPa, n = 18")

# ============================================================
# PRINT MODEL COMPARISON TABLE
# ============================================================
def aic(n,sse,k): return n*np.log(sse/n) + 2*k

print("\n" + "="*72)
print("MODEL COMPARISON TABLE — R², RMSE, AIC")
print("="*72)
print(f"{'Age':<6} {'Model':<28} {'R²':>7} {'RMSE (MPa)':>12} {'AIC':>8} {'':>6}")
print("-"*72)

for age,t in ages_days.items():
    fc = comp_mean[age]; P = por_mean; n_pts = len(fc)

    # R-D
    popt_rd,_ = curve_fit(rd_model, P, fc, p0=[100,0.3])
    yp_rd = rd_model(P,*popt_rd)
    sse_rd = np.sum((fc-yp_rd)**2)

    # Power-law
    popt_pw,_ = curve_fit(pw_model, P, fc, p0=[200,-1])
    yp_pw = pw_model(P,*popt_pw)
    sse_pw = np.sum((fc-yp_pw)**2)

    # Unified (evaluated at this age only)
    yp_un = unified_model((P, np.full(n_pts,float(t))), *popt_uni)
    sse_un = np.sum((fc-yp_un)**2)

    rows = [
        ('R–D (2 params)',       yp_rd, sse_rd, 2),
        ('Power-law (2 params)', yp_pw, sse_pw, 2),
        ('Unified A·tᵅ·Pᵝ',     yp_un, sse_un, 3),
    ]

    # best AIC
    aics = {name: aic(n_pts,sse,k) for name,yp,sse,k in rows}
    best = min(aics, key=aics.get)

    for name,yp,sse,k in rows:
        flag = '✓ BEST' if name==best else ''
        print(f"{age:<6} {name:<28} {r2(fc,yp):>7.4f} {rmse(fc,yp):>12.4f} "
              f"{aic(n_pts,sse,k):>8.2f} {flag:>6}")
    print()

# Unified model global stats
print("-"*72)
print(f"{'ALL':<6} {'Unified A·tᵅ·Pᵝ (global fit)':<28} "
      f"{r2(fc_all,fc_pred_all):>7.4f} {rmse(fc_all,fc_pred_all):>12.4f} "
      f"{'n=18':>8}")
print("="*72)
print(f"\nUnified model equation:")
print(f"  fc(P, t) = {A_opt:.2f} · t^{alpha_opt:.4f} · P^{beta_opt:.4f}")
print(f"  A  = {A_opt:.2f} ± {perr_uni[0]:.2f}  [MPa·days^-α·%^-β]")
print(f"  α  = {alpha_opt:.4f} ± {perr_uni[1]:.4f}  [hydration kinetics exponent]")
print(f"  β  = {beta_opt:.4f} ± {perr_uni[2]:.4f}  [porosity sensitivity exponent]")
print("="*72)