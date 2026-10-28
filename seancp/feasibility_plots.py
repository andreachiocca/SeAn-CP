"""Simple publication-style figures for Phase 5 feasibility comparisons."""
from __future__ import annotations


def plot_reduction(rows,path):
    import matplotlib
    matplotlib.use("Agg",force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    cases=[2213,1739,6920,7348]
    labels=["Exact original","Three normal axes","Plane stress","Tube, signed pressure","Tube, positive p_i"]
    names=["original","three_normal","plane_stress","tube_signed","tube_positive_inner"]
    palette=["#152b45","#327e91","#bf8a3c","#8c67a1","#b2b4b7"]
    fig,ax=plt.subplots(figsize=(10.8,5.8),dpi=150)
    x=np.arange(len(cases));width=.16
    for i,name in enumerate(names):
        y=[-next(r["error_percent"] for r in rows if r["case_id"]==cid and r["model"]==name) for cid in cases]
        bars=ax.bar(x+(i-2)*width,y,width,color=palette[i],label=labels[i])
        for b,v in zip(bars,y):
            if v>=4:ax.text(b.get_x()+b.get_width()/2,v+.6,f"{v:.1f}",ha="center",fontsize=8,rotation=90)
    ax.set_xticks(x,[f"Case {c}" for c in cases]);ax.set_ylabel("SeAn-CP underestimation in FS_ext (%)")
    ax.set_title("Error retained after fitting experimentally simpler stress spaces")
    ax.set_ylim(0,45);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    ax.legend(loc="upper right",fontsize=8,frameon=False,ncol=2)
    fig.text(.1,.01,"The signed-pressure tube fit permits negative gauge pressure and is not a feasible loading program.\n"
             "The positive-pressure fit is at the outer wall only; see CSV for the full-wall check.",fontsize=8)
    fig.tight_layout(rect=(0,.08,1,.99));fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)


def plot_cube_loads(rows,path):
    import matplotlib
    matplotlib.use("Agg",force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax=plt.subplots(figsize=(8.5,4.6),dpi=150)
    cases=[r["case_id"] for r in rows]
    norm=[r["max_face_component_load_kn"] for r in rows]
    tang=[r["max_face_shear_component_load_kn"] for r in rows]
    x=np.arange(len(cases));w=.34
    ax.bar(x-w/2,norm,w,color="#285777",label="Largest force component")
    ax.bar(x+w/2,tang,w,color="#d6a74f",label="Largest tangential component")
    for i,(a,b) in enumerate(zip(norm,tang)):
        ax.text(i-w/2,a+.25,f"{a:.1f}",ha="center",fontsize=9)
        if b>.2:ax.text(i+w/2,b+.25,f"{b:.1f}",ha="center",fontsize=9)
    ax.set_xticks(x,[str(v) for v in cases]);ax.set_ylabel("Absolute peak force component on ONE 5 mm face (kN)")
    ax.set_xlabel("Phase 4 case ID")
    ax.set_title("Ideal uniform-traction cube, illustrative 5 mm side")
    ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(path,dpi=180,bbox_inches="tight");plt.close(fig)
