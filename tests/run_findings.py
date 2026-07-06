import matplotlib
matplotlib.use("Agg")
import pathlib
import matplotlib.pyplot as plt

nbs = ["02_findingD_saddle", "03_findingE_stability",
       "04_findingG_timescale", "05_findingH_subspace"]
for nb in nbs:
    p = pathlib.Path("notebooks") / "two_network" / (nb + ".py")
    print("\n############################", nb)
    g = {"__name__": "__main__"}
    exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"), g)
    for num in plt.get_fignums():
        plt.figure(num).savefig(f"tests/_fig_{nb}.png", dpi=72, bbox_inches="tight")
    plt.close("all")
print("\nALL FINDING NOTEBOOKS RAN")
