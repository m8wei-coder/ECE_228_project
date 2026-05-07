import openroad, odb
from openroad import Design, Tech, Timing
from odb import *
import os
import argparse
from glob import glob
# --- Import additional packages here ---
import numpy as np
from math import ceil
import json




# --- Do not edit except to add additional, optional parameters ---

parser = argparse.ArgumentParser(description="ECE 260C MBFF Clustering")

parser.add_argument(
    '--design',
    type=str,
    help="Your design to load. e.g., 'gcd_v1'",
    required=True
)
parser.add_argument(
    '--output',
    type=str,
    help="Output path (defaults to runs/<design>/clustered.odb)"
)


args = parser.parse_args()
tech = Tech()



print("Loading design...")
design = Design(tech)
tech.readLiberty("pdk/lib/sg13g2_stdcell_typ_1p20V_25C_mbff.lib")

design.readDb(f"designs/{args.design}/design.odb")
# Our design databases already have the MBFF LEF files loaded into them. 
library = design.getDb().getLibs()[0]
    

design.evalTclString(f"source pdk/setRC.tcl")
design.evalTclString(f"read_sdc designs/{args.design}/constraints.sdc")
library = design.getDb().getLibs()[0]
dbu_per_micron = library.getDbUnitsPerMicron()
block = design.getBlock()

print("Performing MBFF clustering...")
# --- Your Code Below ---

# ── Step 1: Collect flip-flops ────────────────────────────────────────────────
def collect_flops(block, design):
    flops = []
    for inst in block.getInsts():
        master = inst.getMaster()
        if not master.isSequential():
            continue
        if design.isBuffer(master) or design.isInverter(master):
            continue
        if not inst.isPlaced():
            continue
        # Exclude MBFFs already in design (they have V2X/V4X/H2V2X in name)
        name = master.getName().lower()
        if any(x in name for x in ['v2x', 'v4x', 'h2v2x']):
            continue
        flops.append(inst)
    return flops

# ── Step 2: Group by (clk_net, rst_net) ──────────────────────────────────────
def group_by_clock_reset(flops, design):
    groups = {}
    for inst in flops:
        clk_net = None
        for iterm in inst.getITerms():
            if design.isInClock(iterm):
                net = iterm.getNet()
                clk_net = net.getName() if net else None
                break
        groups.setdefault(clk_net, []).append(inst)
    return groups

# ── Step 3: Capacity-constrained K-means ─────────────────────────────────────
def cluster_flops_in_group(flop_list, max_cluster_size=4):
    n = len(flop_list)
    if n == 1:
        return [[flop_list[0]]]

    k = ceil(n / max_cluster_size)
    positions = np.array([inst.getLocation() for inst in flop_list], dtype=float)

    # K-means++ initialisation
    rng = np.random.default_rng(42)
    centers = [positions[rng.integers(n)]]
    for _ in range(k - 1):
        dists = np.min([np.sum((positions - c) ** 2, axis=1) for c in centers], axis=0)
        probs = dists / dists.sum()
        centers.append(positions[rng.choice(n, p=probs)])
    centers = np.array(centers)

    for _ in range(100):
        dists = np.linalg.norm(positions[:, None, :] - centers[None, :, :], axis=2)
        # Capacity-constrained assignment: greedily assign each flop to its
        # nearest cluster that still has room.
        order = np.argsort(dists.min(axis=1))  # process most-constrained first
        assignment = [-1] * n
        counts = [0] * k
        for idx in order:
            sorted_clusters = np.argsort(dists[idx])
            for c in sorted_clusters:
                if counts[c] < max_cluster_size:
                    assignment[idx] = c
                    counts[c] += 1
                    break

        # Recompute centroids
        new_centers = np.zeros_like(centers)
        for c in range(k):
            members = [i for i, a in enumerate(assignment) if a == c]
            if members:
                new_centers[c] = positions[members].mean(axis=0)
            else:
                new_centers[c] = centers[c]

        if np.allclose(centers, new_centers, atol=1.0):
            break
        centers = new_centers

    clusters = [[] for _ in range(k)]
    for i, c in enumerate(assignment):
        clusters[c].append(flop_list[i])
    return [cl for cl in clusters if cl]

# ── Step 4: Select MBFF master ────────────────────────────────────────────────
MBFF_MASTER_2BIT = "sg13g2_dfrbpq_V2X_1"
MBFF_MASTER_4BIT = "sg13g2_dfrbpq_V4X_1"

def select_mbff_master(cluster_size, library):
    if cluster_size >= 4:
        return MBFF_MASTER_4BIT
    if cluster_size >= 2:
        return MBFF_MASTER_2BIT
    return None

# ── Step 5: Build cluster list ────────────────────────────────────────────────
def build_clusters(groups, library, max_cluster_size=4):
    clusters = []
    for key, flop_list in groups.items():
        raw_clusters = cluster_flops_in_group(flop_list, max_cluster_size)
        for cl in raw_clusters:
            size = len(cl)
            if size < 2:
                continue
            # Prefer 4-bit when possible, split remainder into 2-bit
            # If size is odd (e.g. 3), use one 2-bit + leave 1 as single
            best_size = 4 if size >= 4 else 2
            # Build sub-clusters of best_size (greedy)
            while len(cl) >= 2:
                take = min(best_size, len(cl))
                if take == 3:
                    take = 2  # avoid 3-bit (unsupported)
                sub = cl[:take]
                cl = cl[take:]
                master_name = select_mbff_master(len(sub), library)
                if master_name is None:
                    continue
                xs = [inst.getLocation()[0] for inst in sub]
                ys = [inst.getLocation()[1] for inst in sub]
                clusters.append({
                    "flops": sub,
                    "mbff_master": master_name,
                    "target_x": float(np.mean(xs)),
                    "target_y": float(np.mean(ys)),
                    "orientation": "N",
                })
    return clusters

# ── Step 6: Sanity check ──────────────────────────────────────────────────────
def sanity_check(clusters, all_flops, design, max_cluster_size=4):
    print("\n── Sanity Check ─────────────────────────────────────────")
    ok = True

    seen = {}
    for ci, cl in enumerate(clusters):
        for inst in cl["flops"]:
            if inst.getName() in seen:
                print(f"  FAIL: {inst.getName()} appears in clusters {seen[inst.getName()]} and {ci}")
                ok = False
            seen[inst.getName()] = ci

    for ci, cl in enumerate(clusters):
        if len(cl["flops"]) > max_cluster_size:
            print(f"  FAIL: cluster {ci} has size {len(cl['flops'])} > {max_cluster_size}")
            ok = False

    def get_clk(inst):
        for iterm in inst.getITerms():
            if design.isInClock(iterm):
                net = iterm.getNet()
                return net.getName() if net else None
        return None

    for ci, cl in enumerate(clusters):
        clks = set(get_clk(inst) for inst in cl["flops"])
        if len(clks) > 1:
            print(f"  FAIL: cluster {ci} has mixed CLK nets: {clks}")
            ok = False

    print(f"  {'PASS' if ok else 'FAIL'}: all checks done")
    print("─────────────────────────────────────────────────────────\n")
    return ok

# ── Step 7: QoR summary ───────────────────────────────────────────────────────
def print_qor(clusters, all_flops):
    total = len(all_flops)
    n2 = sum(1 for cl in clusters if len(cl["flops"]) == 2)
    n4 = sum(1 for cl in clusters if len(cl["flops"]) == 4)
    f2 = n2 * 2
    f4 = n4 * 4
    clustered = sum(len(cl["flops"]) for cl in clusters)
    single = total - clustered
    print("══ QoR Summary ══════════════════════════════════════════")
    print(f"  Total flops          : {total}")
    print(f"  Total clusters       : {len(clusters)}")
    print(f"  2-bit clusters       : {n2}  ({f2} flops)")
    print(f"  4-bit clusters       : {n4}  ({f4} flops)")
    print(f"  Unchanged singles    : {single}")
    print(f"  Cluster ratio        : {clustered/total:.2%}" if total else "  Cluster ratio: N/A")
    print("═════════════════════════════════════════════════════════\n")

# ── Main ──────────────────────────────────────────────────────────────────────
all_flops = collect_flops(block, design)
print(f"Found {len(all_flops)} single-bit flip-flops to cluster.")

groups = group_by_clock_reset(all_flops, design)
print(f"Grouped into {len(groups)} (clk, rst) groups.")

clusters = build_clusters(groups, library, max_cluster_size=4)

sanity_check(clusters, all_flops, design, max_cluster_size=4)
print_qor(clusters, all_flops)

clusters_json_path = f"runs/{args.design}/clusters.json"
os.makedirs(f"runs/{args.design}", exist_ok=True)
serializable = [
    {
        "flop_names": [inst.getName() for inst in cl["flops"]],
        "mbff_master": cl["mbff_master"],
        "target_x": cl["target_x"],
        "target_y": cl["target_y"],
        "orientation": cl["orientation"],
    }
    for cl in clusters
]
with open(clusters_json_path, "w") as f:
    json.dump(serializable, f, indent=2)
print(f"Cluster assignments written to {clusters_json_path}")





# --- Do not edit ---
print("Writing Database...")

output_path = args.output if args.output else f"runs/{args.design}"

os.makedirs(output_path, exist_ok=True)

design.writeDb(f"{output_path}/clustered.odb")
print(f"Wrote to {output_path}/clustered.odb")