"""Diagnostic FS_ext angle landscapes with all three SeAn-CP search paths."""
from pathlib import Path
import numpy as np
from .core import evaluate_fs_ext,full_plane_scan,seancp_fs_ext

COLORS={"strain_range":"#e6bf4b","stress_anchor_0":"#f36b4b",
        "stress_anchor_1":"#c57af0"}
STYLES={"strain_range":"-","stress_anchor_0":"--","stress_anchor_1":":"}


def pyplot():
    import matplotlib
    matplotlib.use("Agg",force=True)
    import matplotlib.pyplot as plt
    return plt


def hemisphere(normals):
    """Use the upper hemisphere because n and -n describe the same plane."""
    n=np.asarray(normals,float).copy()
    flip=n[:,2]<-1e-12
    equator=np.abs(n[:,2])<=1e-12
    flip |= equator & ((n[:,0]<-1e-12) |
                         ((np.abs(n[:,0])<=1e-12) & (n[:,1]<0)))
    n[flip]*=-1
    n[:,2]=np.maximum(n[:,2],0)
    theta=np.degrees(np.arccos(np.clip(n[:,2],-1,1)))
    phi=np.degrees(np.arctan2(n[:,1],n[:,0])) % 360
    return phi,theta


def segments(phi,theta,values):
    seam=np.flatnonzero(np.abs(np.diff(phi))>180)+1
    for idx in np.split(np.arange(len(phi)),seam):
        if len(idx)>1:
            yield phi[idx],theta[idx],values[idx]


def plot_damage_landscape(history,config,outfile: str | Path,*,
                          title:str="FS_ext plane landscape",
                          algorithm:str="longest_chord",three_d:bool=False):
    """Full surface and three restricted paths on a hemisphere of plane normals."""
    plt=pyplot()
    theta=np.linspace(0,np.pi/2,121)
    phi=np.linspace(0,2*np.pi,241)
    T,P=np.meshgrid(theta,phi,indexing="ij")
    N=np.stack((np.sin(T)*np.cos(P),np.sin(T)*np.sin(P),np.cos(T)),axis=-1).reshape(-1,3)
    damage=evaluate_fs_ext(history,N,k_fs=config.k_fs,sigma_y=config.sigma_y_mpa,
                           algorithm=algorithm).reshape(T.shape)
    semi=seancp_fs_ext(history,k_fs=config.k_fs,sigma_y=config.sigma_y_mpa,
                       algorithm=algorithm,samples_per_path=361)
    full,nref=full_plane_scan(history,k_fs=config.k_fs,sigma_y=config.sigma_y_mpa,
                             algorithm=algorithm,n_theta=181,n_phi=361)
    pref,tref=hemisphere(nref[None,:]);psemi,tsemi=hemisphere(semi.normal[None,:])
    fig=plt.figure(figsize=(10.2,7.0 if three_d else 5.8),dpi=130)
    if three_d:
        ax=fig.add_subplot(111,projection="3d")
        mesh=ax.plot_surface(np.degrees(P),np.degrees(T),1000*damage,cmap="viridis",
                             alpha=.74,rstride=2,cstride=2,linewidth=0,shade=False)
        margin=max(float(np.max(damage))*.07,1e-9)*1000
        for path in semi.paths:
            pp,tt=hemisphere(path.normals)
            for x,y,z in segments(pp,tt,1000*path.values+margin):
                ax.plot(x,y,z,color=COLORS[path.family],lw=2.8,
                        linestyle=STYLES[path.family])
        ax.scatter(pref,tref,[1000*full+2*margin],s=160,c="white",marker="*",
                   edgecolors="#14283e",depthshade=False)
        ax.scatter(psemi,tsemi,[1000*semi.value+2*margin],s=125,marker="o",
                   facecolors="none",edgecolors="#101923",linewidths=2,depthshade=False)
        ax.set_zlabel("1000 × FS_ext")
        ax.view_init(elev=29,azim=-55)
        ax.set_xlim(0,360);ax.set_ylim(0,90)
        fig.colorbar(mesh,ax=ax,shrink=.6,pad=.11,label="1000 × FS_ext")
        fig.text(.09,.075,"White star: full plane | Open circle: SeAn-CP",fontsize=9)
    else:
        ax=fig.add_subplot(111)
        mesh=ax.pcolormesh(np.degrees(phi),np.degrees(theta),1000*damage,
                           cmap="viridis",shading="auto")
        for path in semi.paths:
            pp,tt=hemisphere(path.normals)
            for idx,(x,y,z) in enumerate(segments(pp,tt,path.values)):
                ax.plot(x,y,color=COLORS[path.family],lw=2.5,
                        linestyle=STYLES[path.family],
                        label=path.family if idx==0 else None)
        ax.scatter(pref,tref,s=175,c="white",edgecolor="#14283e",marker="*",
                   linewidth=1.4,zorder=7,label="Full plane")
        ax.scatter(psemi,tsemi,s=115,facecolors="none",edgecolors="#111111",
                   marker="o",linewidth=2.1,zorder=7,label="SeAn-CP")
        ax.set_xticks(np.arange(0,361,60));ax.set_yticks(np.arange(0,91,15))
        ax.grid(alpha=.16,color="white")
        fig.colorbar(mesh,ax=ax,label="1000 × FS_ext")
        ax.legend(loc="upper center",bbox_to_anchor=(.5,-.16),ncol=5,
                  frameon=False,fontsize=8)
    ax.set_xlabel("Azimuth φ (degrees)")
    ax.set_ylabel("Polar angle θ (degrees)")
    error=100*(semi.value/full-1) if full else 0
    fig.suptitle(f"{title}\nFull {full:.7g} | SeAn-CP {semi.value:.7g} | error {error:+.3f}%")
    if not three_d: fig.tight_layout(rect=(0,.02,1,.87))
    else: fig.subplots_adjust(top=.84,bottom=.08)
    fig.savefig(outfile,dpi=160,bbox_inches="tight")
    plt.close(fig)


def plot_screening_summary(rows,outfile):
    plt=pyplot()
    families=("control","proportional","torque_90","pressure_90",
              "mixed_90","general","mean_shifted")
    fig,ax=plt.subplots(figsize=(10,5.5),dpi=140)
    rng=np.random.default_rng(17)
    for i,family in enumerate(families):
        for offset,alg,color in ((-.12,"longest_chord","#296c9e"),(.12,"mce","#c17b3c")):
            errors=[x["error_wall_percent"] for x in rows
                    if x["family"]==family and x["algorithm"]==alg
                    and np.isfinite(x["error_wall_percent"])]
            if errors:
                ax.scatter(errors,i+offset+rng.uniform(-.06,.06,len(errors)),s=13,
                           color=color,alpha=.53,label=alg if i==0 else None)
                ax.plot(np.median(errors),i+offset,"D",color=color,ms=5)
    ax.set_yticks(range(len(families)),labels=families)
    ax.set_xlabel("Signed wall-wide FS_ext error (%)")
    ax.set_ylabel("Harmonic load family")
    ax.axvline(0,color="#222222",lw=1)
    ax.grid(axis="x",alpha=.18)
    ax.invert_yaxis();ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(outfile,dpi=160,bbox_inches="tight");plt.close(fig)


def plot_radial_comparison(rows,outfile,*,title=""):
    plt=pyplot();fig,ax=plt.subplots(figsize=(8,4.7),dpi=140)
    for alg,color in (("longest_chord","#296c9e"),("mce","#c17b3c")):
        data=sorted((x for x in rows if x["algorithm"]==alg),key=lambda x:x["radius_mm"])
        if not data: continue
        ax.plot([x["radius_mm"] for x in data],
                [1000*x["fs_ext_reference"] for x in data],color=color,lw=2,
                label=f"Full plane, {alg}")
        ax.plot([x["radius_mm"] for x in data],
                [1000*x["fs_ext_seancp"] for x in data],color=color,lw=2,ls="--",
                label=f"SeAn-CP, {alg}")
    ax.set_xlabel("Radius (mm)");ax.set_ylabel("1000 × FS_ext")
    ax.set_title("Through-wall comparison | "+title)
    ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(outfile,dpi=160,bbox_inches="tight");plt.close(fig)
