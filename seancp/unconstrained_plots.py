"""Reproducible 3D FS_ext damage maps with the actual SeAn-CP trajectories.

Coordinates: azimuth [degrees] on x, polar angle [degrees] on y,
1000 x FS_ext on z. Unit normals n and -n represent the same plane.
"""
from pathlib import Path
import numpy as np

from .core import evaluate_fs_ext, full_plane_scan, seancp_fs_ext
from .study_plots import hemisphere, segments

PATH_COLORS={"strain_range":"#e3b94d", "stress_anchor_0":"#ce5878",
             "stress_anchor_1":"#5489d4"}
PATH_STYLES={"strain_range":"-", "stress_anchor_0":"--",
             "stress_anchor_1":":"}


def plot_comparison(history,config,outfile: str | Path,*,label:str,
                    algorithm:str="mce",three_d:bool=True):
    """Plot maximum FS_ext and all three corrected semi-analytical paths.

    Both maximum markers and each path use their true damage values. There
    is no arbitrary vertical elevation of the curves or markers.
    """
    import matplotlib
    matplotlib.use("Agg",force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    kw=dict(k_fs=config.k_fs,sigma_y=config.sigma_y_mpa,algorithm=algorithm)
    theta=np.linspace(0,np.pi/2,121)
    phi=np.linspace(0,2*np.pi,241)
    T,P=np.meshgrid(theta,phi,indexing="ij")
    normals=np.stack((np.sin(T)*np.cos(P),np.sin(T)*np.sin(P),np.cos(T)),axis=-1).reshape(-1,3)
    damage=evaluate_fs_ext(history,normals,**kw).reshape(T.shape)
    semi=seancp_fs_ext(history,samples_per_path=721,**kw)
    full,full_normal=full_plane_scan(history,n_theta=181,n_phi=361,**kw)
    pmax,tmax=hemisphere(full_normal[None,:])
    psemi,tsemi=hemisphere(semi.normal[None,:])
    x=np.rad2deg(P);y=np.rad2deg(T);z=1e3*damage
    error=100*(semi.value/full-1)

    if three_d:
        fig=plt.figure(figsize=(11.6,8),dpi=140,facecolor="white")
        ax=fig.add_subplot(111,projection="3d")
        surface=ax.plot_surface(x,y,z,cmap="viridis",alpha=.48,
                                rstride=2,cstride=2,linewidth=0,
                                antialiased=True,shade=False)
        # Each segment is evaluated at the *actual* damage of its path.
        # Seam splitting prevents artificial connecting chords across phi=0/360.
        for path in semi.paths:
            xx,yy=hemisphere(path.normals)
            for u,v,w in segments(xx,yy,1000*path.values):
                ax.plot(u,v,w,linewidth=3.0,color=PATH_COLORS[path.family],
                        linestyle=PATH_STYLES[path.family],alpha=.99)
        ax.scatter(pmax,tmax,[1e3*full],marker="*",s=235,depthshade=False,
                   color="#fefefe",edgecolors="#202020",linewidth=1.8,zorder=100)
        ax.scatter(psemi,tsemi,[1e3*semi.value],marker="o",s=175,depthshade=False,
                   color="#f6d463",edgecolors="#272525",linewidth=2.0,zorder=101)
        ax.view_init(elev=27,azim=-57)
        ax.set_box_aspect((1.3,1,.7),zoom=.92)
        ax.set_xlim(0,360);ax.set_ylim(0,90)
        ax.set_xticks([0,60,120,180,240,300,360]);ax.set_yticks([0,30,60,90])
        ax.set_zlabel(r"$10^3\,FS_{ext}$",labelpad=12)
        ax.set_xlabel("Azimuth φ (°)",labelpad=10)
        ax.set_ylabel("Polar angle θ (°)",labelpad=10)
        ax.xaxis.pane.fill = False;ax.yaxis.pane.fill = False;ax.zaxis.pane.fill = False
        cb=fig.colorbar(surface,ax=ax,shrink=.53,pad=.09)
        cb.set_label(r"$10^3\,FS_{ext}$")
        bottom=.12
    else:
        fig,ax=plt.subplots(figsize=(10.6,6.8),dpi=140)
        hm=ax.pcolormesh(np.rad2deg(phi),np.rad2deg(theta),z,
                         shading="auto",cmap="viridis")
        for path in semi.paths:
            xx,yy=hemisphere(path.normals)
            for u,v,_ in segments(xx,yy,path.values):
                ax.plot(u,v,color=PATH_COLORS[path.family],lw=2.7,
                        linestyle=PATH_STYLES[path.family])
        ax.scatter(pmax,tmax,marker="*",s=210,c="white",edgecolor="#202020",linewidth=1.4,zorder=10)
        ax.scatter(psemi,tsemi,marker="o",s=120,c="#f6d463",edgecolor="#272525",linewidth=1.7,zorder=11)
        ax.set(xlim=(0,360),ylim=(0,90),xlabel="Azimuth φ (°)",ylabel="Polar angle θ (°)")
        ax.set_xticks([0,60,120,180,240,300,360]);ax.set_yticks([0,15,30,45,60,75,90])
        fig.colorbar(hm,ax=ax,label=r"$10^3\,FS_{ext}$")
        bottom=.18
    legend=[Line2D([0],[0],lw=2.8,color=PATH_COLORS[name],linestyle=PATH_STYLES[name],
                   label={"strain_range":"Strain-range path", "stress_anchor_0":"Stress path 1",
                          "stress_anchor_1":"Stress path 2"}[name])
            for name in PATH_COLORS]
    legend += [Line2D([0],[0],marker="o",ms=10,lw=0,mfc="#f6d463",mec="#202020",label="SeAn-CP best"),
               Line2D([0],[0],marker="*",ms=15,lw=0,mfc="white",mec="#202020",label="Full-plane maximum")]
    fig.legend(handles=legend,loc="lower center",bbox_to_anchor=(.5,.045),
               ncol=5,frameon=False,fontsize=9)
    fig.suptitle(f"{label} | {algorithm}\nFull FS_ext = {full:.7f}  |  SeAn-CP = {semi.value:.7f}  |  error = {error:+.2f}%",
                 y=.97,fontsize=12)
    if three_d:
        fig.subplots_adjust(left=.02,right=.9,top=.86,bottom=bottom)
    else:
        fig.tight_layout(rect=(.02,bottom,.98,.84))
    fig.savefig(outfile,dpi=170,bbox_inches="tight")
    plt.close(fig)
    return dict(full=full,semi=semi.value,error_percent=error,
                full_normal=full_normal.tolist(),semi_normal=semi.normal.tolist())


def plot_summary(rows,outfile):
    import matplotlib
    matplotlib.use("Agg",force=True)
    import matplotlib.pyplot as plt
    ordered=sorted(rows,key=lambda x:x["error_percent"])
    fig,ax=plt.subplots(figsize=(9,4.5),dpi=160)
    vals=[x["error_percent"] for x in ordered]
    ax.barh([x["family"].replace("_"," ") for x in ordered],
            [-v for v in vals],color="#346f9a")
    ax.invert_yaxis()
    for i,v in enumerate(vals):ax.text(-v+.4,i,f"{-v:.2f}%",va="center",fontsize=10)
    ax.set_xlabel("FS_ext underestimation relative to full-plane search (%)")
    ax.set_xlim(0,max(-v for v in vals)*1.18)
    ax.set_title("Four distinct harmonic tensor loading families")
    ax.grid(axis="x",alpha=.15)
    ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(outfile,dpi=180,bbox_inches="tight")
    plt.close(fig)
