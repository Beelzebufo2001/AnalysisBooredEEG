"""
EEG Bridge Detector — Graph-Theory Based
=========================================
Improved bridge detection for dense montages (1005, 1010).
Replaces MNE's pairwise correlation approach with:
  - Distance-conditioned residual correlation
  - Connected component clustering (gel blobs, not edge lists)
  - Spatial chain validation (continuity enforcement)
  - Triangle consistency pruning (no phantom long-range bridges)
  - Temporal slope scoring (gel settling vs stable anatomy)

Dependencies:
    pip install mne numpy scipy networkx matplotlib scikit-learn

Usage:
    detector = BridgeDetector(raw, montage="standard_1005")
    clusters = detector.run()
    detector.plot_clusters(clusters)
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import mne

from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from scipy.signal import correlate
from sklearn.linear_model import HuberRegressor
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────

@dataclass
class BridgeCluster: #AKA gel blop class with return information 
    """
    One physical gel blob = one BridgeCluster.
    Not a list of edges — an object with spatial and temporal properties.
    """
    electrodes:       list[str]           # electrode names in this cluster
    centroid:         np.ndarray          # mean 3D position
    spatial_diameter: float               # max pairwise distance in cluster (mm)
    density_score:    float               # mean residual correlation inside cluster
    temporal_slope:   Optional[float]     # +ve = gel forming, ~0 = anatomical, None = not computed
    confidence:       float               # composite score [0–1]
    raw_edges:        list[tuple]         # (ch_i, ch_j, residual_corr) for inspection

    def __repr__(self): #return info about blop
        slopes = f"{self.temporal_slope:+.3f}" if self.temporal_slope is not None else "n/a"
        return (
            f"BridgeCluster({len(self.electrodes)} electrodes | "
            f"confidence={self.confidence:.2f} | "
            f"diameter={self.spatial_diameter:.1f}mm | "
            f"temp_slope={slopes})\n"
            f"  electrodes: {self.electrodes}"
        )


# ─────────────────────────────────────────────
#  Core detector
# ─────────────────────────────────────────────

class BridgeDetector:

    def __init__( #This shit constructor
        self,
        raw: mne.io.BaseRaw,
        montage: str = "standard_1005",
        
        # Correlation window (seconds) — how long a segment for correlation computation
        window_sec: float = 60.0,
        
        # How much of the BEGINNING of recording to use for temporal slope check
        early_window_sec: float = 180.0,
        
        # Top-k percentile of residual edges to keep (montage-density invariant)
        edge_percentile: float = 97.5,
        
        # Z-score threshold on residuals (used if percentile gives too many edges)
        residual_z_threshold: float = 2.5,
        
        # Triangle consistency: if (i,j) and (j,k) are edges, (i,k) residual must be >= this fraction of mean(ij,jk)
        triangle_consistency_ratio: float = 0.4, """ mne angorithm would flag it which mean that ij or jk could be false positive or ik is false negative... which we can only guess """
        
        # Spatial continuity: reject edge (i,j) if no intermediate electrode within this fraction of i-j distance
        chain_gap_ratio: float = 0.6,
        
        # Minimum electrodes to report a cluster
        min_cluster_size: int = 2,
    ):
        self.raw = raw.copy().pick_types(eeg=True)
        self.sfreq = raw.info["sfreq"]
        self.ch_names = self.raw.ch_names

        self.window_sec         = window_sec
        self.early_window_sec   = early_window_sec
        self.edge_percentile    = edge_percentile
        self.residual_z_thresh  = residual_z_threshold
        self.triangle_ratio     = triangle_consistency_ratio
        self.chain_gap_ratio    = chain_gap_ratio
        self.min_cluster_size   = min_cluster_size

        # Set montage and extract 3D positions
        self.raw.set_montage(montage, on_missing="warn")
        self.positions = self._get_positions()  # shape (n_ch, 3), mm units

        self.n_ch = len(self.ch_names)
        print(f"[BridgeDetector] {self.n_ch} EEG channels loaded.")

    # ─── Public API ───────────────────────────────────────────

    def run(self, verbose: bool = True) -> list[BridgeCluster]:
        """Full detection pipeline. Returns sorted list of BridgeCluster objects."""

        if verbose: print("[1/7] Computing scalp distances...")
        dist_matrix = self._scalp_distances()                          # (n, n) mm

        if verbose: print("[2/7] Computing correlation matrix...")
        corr_matrix = self._correlation_matrix()                       # (n, n)

        if verbose: print("[3/7] Fitting distance→correlation model (residuals)...")
        residual_matrix = self._residual_correlations(corr_matrix, dist_matrix)

        if verbose: print("[4/7] Building sparse residual graph...")
        G = self._build_graph(residual_matrix, dist_matrix)

        if verbose: print("[5/7] Pruning spatially impossible edges...")
        G = self._prune_spatial_chains(G, dist_matrix)

        if verbose: print("[6/7] Pruning triangle-inconsistent edges...")
        G = self._prune_triangle_inconsistency(G, residual_matrix)

        if verbose: print("[7/7] Extracting clusters + scoring...")
        clusters = self._extract_clusters(G, residual_matrix, dist_matrix)

        if verbose:
            print(f"\n{'─'*55}")
            print(f"  Found {len(clusters)} bridge cluster(s)")
            print(f"{'─'*55}")
            for c in clusters:
                print(c)

        return clusters

    def plot_clusters(self, clusters: list[BridgeCluster], top_n: int = 10):
        """Plot bridge clusters on a topomap-style 2D scalp projection."""
        if not clusters:
            print("No clusters to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 9))
        ax.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#0d1117")
        ax.set_title("Bridge Clusters (top N by confidence)", color="white", fontsize=13)

        # All electrode positions projected to 2D (x/y from 3D, ignoring z)
        pos2d = {ch: self.positions[i, :2] for i, ch in enumerate(self.ch_names)}

        # Draw all electrodes as small dots
        for ch, (x, y) in pos2d.items():
            ax.scatter(x, y, s=10, c="#444", zorder=1)

        colors = plt.cm.plasma(np.linspace(0.2, 0.95, min(top_n, len(clusters))))

        for idx, (cluster, color) in enumerate(zip(clusters[:top_n], colors)):
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(color[0]*255), int(color[1]*255), int(color[2]*255)
            )
            for ch in cluster.electrodes:
                if ch in pos2d:
                    x, y = pos2d[ch]
                    ax.scatter(x, y, s=90, c=hex_color, zorder=3, edgecolors="white", linewidths=0.5)
                    ax.text(x, y + 1.5, ch, fontsize=5, color="white", ha="center", zorder=4)

            for (chi, chj, _) in cluster.raw_edges:
                if chi in pos2d and chj in pos2d:
                    xi, yi = pos2d[chi]
                    xj, yj = pos2d[chj]
                    ax.plot([xi, xj], [yi, yj], c=hex_color, alpha=0.6, lw=1.2, zorder=2)

            cx, cy = cluster.centroid[:2]
            ax.text(cx, cy - 3, f"#{idx+1} conf={cluster.confidence:.2f}",
                    fontsize=7, color=hex_color, ha="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#0d1117", ec=hex_color, alpha=0.7))

        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    # ─── Step implementations ──────────────────────────────────

    def _get_positions(self) -> np.ndarray:
        """Extract 3D electrode positions in mm."""
        montage = self.raw.get_montage()
        if montage is None:
            raise ValueError("No montage set. Cannot compute scalp distances.")
        pos_dict = {
            ch: montage.get_positions()["ch_pos"][ch] * 1000  # m → mm
            for ch in self.ch_names
            if ch in montage.get_positions()["ch_pos"]
        }
        # Align ordering to self.ch_names, drop missing
        self.ch_names = [ch for ch in self.ch_names if ch in pos_dict]
        return np.array([pos_dict[ch] for ch in self.ch_names])

    def _scalp_distances(self) -> np.ndarray:
        """
        Geodesic scalp distance approximation via arc-length on unit sphere.
        Better than Euclidean for curved scalp, cheap to compute.
        Returns (n_ch, n_ch) matrix in mm.
        """
        # Normalize to unit sphere
        norms = np.linalg.norm(self.positions, axis=1, keepdims=True)
        unit = self.positions / (norms + 1e-9)

        # Arc length: d = R * arccos(u_i · u_j)
        dots = np.clip(unit @ unit.T, -1.0, 1.0)
        R = np.mean(norms)  # mean head radius in mm
        return R * np.arccos(dots)

    def _correlation_matrix(self) -> np.ndarray:
        """
        Pearson correlation on a stable window of data.
        Uses middle portion of recording to avoid edge artifacts.
        """
        data = self.raw.get_data(units="uV")  # (n_ch, n_times)

        n_times = data.shape[1]
        win_samples = int(self.window_sec * self.sfreq)
        win_samples = min(win_samples, n_times)

        # Use middle segment — most settled state
        start = max(0, (n_times - win_samples) // 2)
        segment = data[:, start: start + win_samples]

        # Z-score each channel before correlation (removes amplitude scale bias)
        segment = zscore(segment, axis=1)
        corr = np.corrcoef(segment)
        np.fill_diagonal(corr, 0.0)
        return corr

    def _residual_correlations(
        self, corr: np.ndarray, dist: np.ndarray
    ) -> np.ndarray:
        """
        Fit a distance-decay model: E[corr | distance].
        Residual = observed − expected.
        This removes the 'nearby electrodes are naturally similar' confound.

        Model: corr ~ a * exp(-b * dist) + c   (or linear in log space)
        Uses Huber regression for robustness against outlier bridges.
        """
        n = corr.shape[0]
        upper_idx = np.triu_indices(n, k=1)
        d_flat = dist[upper_idx].reshape(-1, 1)
        c_flat = corr[upper_idx]

        # Features: [dist, dist^2, 1/dist] — flexible decay without strict exp assumption
        X = np.hstack([
            d_flat,
            d_flat**2,
            1.0 / (d_flat + 1.0),
        ])

        model = HuberRegressor(epsilon=1.5, max_iter=500)
        model.fit(X, c_flat)
        c_expected_flat = model.predict(X)

        residual_flat = c_flat - c_expected_flat

        # Rebuild symmetric matrix
        residual = np.zeros((n, n))
        residual[upper_idx] = residual_flat
        residual += residual.T  # symmetrize
        np.fill_diagonal(residual, 0.0)
        return residual

    def _build_graph(
        self, residual: np.ndarray, dist: np.ndarray
    ) -> nx.Graph:
        """
        Build sparse graph from top-percentile residual edges.
        Density-invariant: keeps top X% globally, not fixed threshold.
        """
        n = residual.shape[0]
        upper_idx = np.triu_indices(n, k=1)
        r_flat = residual[upper_idx]

        threshold = np.percentile(r_flat, self.edge_percentile)
        # Also enforce a minimum z-score
        z_flat = (r_flat - r_flat.mean()) / (r_flat.std() + 1e-9)
        z_threshold = self.residual_z_thresh

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for idx, (i, j) in enumerate(zip(*upper_idx)):
            if r_flat[idx] >= threshold and z_flat[idx] >= z_threshold:
                G.add_edge(i, j,
                           residual=float(r_flat[idx]),
                           distance=float(dist[i, j]),
                           z_score=float(z_flat[idx]))
        return G

    def _prune_spatial_chains(
        self, G: nx.Graph, dist: np.ndarray
    ) -> nx.Graph:
        """
        Remove edges (i, j) where no intermediate electrode exists
        between them on the scalp — physically impossible gel bridge.

        Rule: for edge (i,j) with distance d_ij,
              there must exist electrode k (not i,j) such that
              dist(i,k) < chain_gap_ratio * d_ij
              AND dist(k,j) < chain_gap_ratio * d_ij
        """
        to_remove = []

        for (i, j) in list(G.edges()):
            d_ij = dist[i, j]
            max_gap = self.chain_gap_ratio * d_ij

            # Is there any intermediate electrode?
            has_intermediate = False
            for k in range(self.n_ch):
                if k == i or k == j:
                    continue
                if dist[i, k] < max_gap and dist[k, j] < max_gap:
                    has_intermediate = True
                    break

            # If no intermediate exists AND edge is long, remove
            # (short edges between true neighbors are always allowed)
            median_dist = np.median(dist[dist > 0])
            if not has_intermediate and d_ij > median_dist * 0.8:
                to_remove.append((i, j))

        G.remove_edges_from(to_remove)
        print(f"    Spatial chain pruning removed {len(to_remove)} edges.")
        return G

    def _prune_triangle_inconsistency(
        self, G: nx.Graph, residual: np.ndarray
    ) -> nx.Graph:
        """
        Triangle consistency: if (i,j) and (j,k) are flagged,
        then (i,k) should show elevated residual too (gel propagates locally).

        If edge (i,j) exists but has no triangles supporting it
        (none of its neighbors share elevated residual with each other),
        it's a suspicious isolated spike → remove.
        """
        to_remove = []

        for (i, j) in list(G.edges()):
            neighbors_i = set(G.neighbors(i)) - {j}
            neighbors_j = set(G.neighbors(j)) - {i}
            shared = neighbors_i & neighbors_j  # nodes that form triangles with (i,j)

            if not shared:
                # No triangle support — check if the residual is extreme enough to keep anyway
                r_ij = residual[i, j]
                r_all = residual[np.triu_indices(residual.shape[0], k=1)]
                if r_ij < np.percentile(r_all, 99.0):  # keep only very extreme isolated edges
                    to_remove.append((i, j))
                continue

            # Check that shared neighbors have consistent residuals
            consistent = False
            for k in shared:
                r_ik = residual[i, k]
                r_jk = residual[j, k]
                r_ij = residual[i, j]
                expected_ik = self.triangle_ratio * r_ij
                if r_ik >= expected_ik or r_jk >= expected_ik:
                    consistent = True
                    break

            if not consistent:
                to_remove.append((i, j))

        G.remove_edges_from(to_remove)
        print(f"    Triangle consistency pruning removed {len(to_remove)} edges.")
        return G

    def _extract_clusters(
        self,
        G: nx.Graph,
        residual: np.ndarray,
        dist: np.ndarray,
    ) -> list[BridgeCluster]:
        """
        Extract connected components → one cluster per gel blob.
        Score each cluster and optionally compute temporal slope.
        """
        clusters = []
        components = [
            c for c in nx.connected_components(G)
            if len(c) >= self.min_cluster_size
        ]

        for comp in components:
            idx_list = sorted(comp)
            names = [self.ch_names[i] for i in idx_list]
            pos = self.positions[idx_list]

            # Internal edges of this cluster
            internal_edges = [
                (self.ch_names[i], self.ch_names[j], residual[i, j])
                for i, j in G.edges()
                if i in comp and j in comp
            ]

            # Density = mean residual correlation among cluster members
            if len(internal_edges) > 0:
                density = np.mean([e[2] for e in internal_edges])
            else:
                density = 0.0

            # Spatial diameter
            if len(idx_list) > 1:
                sub_dist = dist[np.ix_(idx_list, idx_list)]
                diameter = float(np.max(sub_dist))
            else:
                diameter = 0.0

            centroid = pos.mean(axis=0)

            # Temporal slope (gel settling signature)
            temp_slope = self._temporal_slope(idx_list)

            # Confidence score
            # High density + small diameter + positive slope → high confidence bridge
            norm_density = np.tanh(density * 3)            # 0→1
            norm_diameter = np.exp(-diameter / 30.0)       # smaller = better
            if temp_slope is not None:
                norm_slope = np.clip(temp_slope * 10, 0, 1) # positive slope = gel forming
            else:
                norm_slope = 0.5  # neutral if not computed

            confidence = float(
                0.5 * norm_density +
                0.3 * norm_diameter +
                0.2 * norm_slope
            )

            clusters.append(BridgeCluster(
                electrodes=names,
                centroid=centroid,
                spatial_diameter=diameter,
                density_score=float(density),
                temporal_slope=temp_slope,
                confidence=confidence,
                raw_edges=internal_edges,
            ))

        # Sort by confidence descending
        clusters.sort(key=lambda c: c.confidence, reverse=True)
        return clusters

    def _temporal_slope(self, idx_list: list[int]) -> Optional[float]:
        """
        Compute slope of mean pairwise correlation over early recording window.

        Returns:
            positive slope → correlation increasing = likely gel forming (bridge)
            ~zero slope    → stable = likely anatomical similarity (not a bridge)
            negative slope → decreasing = gel drying / noise
            None           → not enough data
        """
        data = self.raw.get_data(units="uV")
        total_samples = data.shape[1]
        early_samples = int(self.early_window_sec * self.sfreq)
        early_samples = min(early_samples, total_samples)

        if early_samples < int(30 * self.sfreq):  # need at least 30s
            return None

        # Sliding windows of 10s each
        win = int(10 * self.sfreq)
        n_windows = early_samples // win
        if n_windows < 3:
            return None

        slopes_input = []
        mean_corrs = []

        for w in range(n_windows):
            seg = data[np.ix_(idx_list, range(w * win, (w + 1) * win))]
            seg = zscore(seg, axis=1)
            if seg.shape[0] < 2:
                mean_corrs.append(0.0)
            else:
                c = np.corrcoef(seg)
                upper = c[np.triu_indices(c.shape[0], k=1)]
                mean_corrs.append(float(np.mean(upper)))
            slopes_input.append(w)

        if len(mean_corrs) < 2:
            return None

        # Simple linear regression slope
        x = np.array(slopes_input, dtype=float)
        y = np.array(mean_corrs, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        return slope


# ─────────────────────────────────────────────
#  Quick usage example
# ─────────────────────────────────────────────

if __name__ == "__main__":
    """
    Example: load a raw EEG file and run the detector.
    Swap the file path + montage to match your setup.
    """

    # Load your file — any MNE-readable format works
    # raw = mne.io.read_raw_brainvision("your_file.vhdr", preload=True)
    # raw = mne.io.read_raw_edf("your_file.edf", preload=True)

    # ── For quick testing: generate synthetic data with a fake bridge ──
    print("Generating synthetic EEG for demo...")
    info = mne.create_info(
        ch_names=mne.channels.make_standard_montage("standard_1020").ch_names[:32],
        sfreq=256.0,
        ch_types="eeg",
    )
    rng = np.random.default_rng(42)
    n_ch, n_t = 32, 256 * 120  # 2 minutes
    data = rng.standard_normal((n_ch, n_t)) * 20e-6  # 20 µV baseline

    # Inject a fake bridge: channels 5 and 6 become very similar
    bridge_signal = rng.standard_normal(n_t) * 20e-6
    data[5] += bridge_signal * 0.95
    data[6] += bridge_signal * 0.95

    raw = mne.io.RawArray(data, info)

    detector = BridgeDetector(
        raw,
        montage="standard_1020",
        window_sec=60.0,
        early_window_sec=120.0,
        edge_percentile=95.0,
        residual_z_threshold=2.0,
        triangle_consistency_ratio=0.4,
        chain_gap_ratio=0.6,
        min_cluster_size=2,
    )

    clusters = detector.run(verbose=True)
    detector.plot_clusters(clusters)