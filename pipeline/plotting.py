"""Display helpers for the CV+ manuscript plots."""
import numpy as np

def violin_points(ax, x, distribution, color, width):
    """Smoothed observed distribution, individual mutations, and group means."""
    artists = ax.violinplot(distribution, positions=x, widths=width,
                           showmeans=False, showmedians=False, showextrema=False, points=150)
    for body in artists['bodies']:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(.25)
        body.set_linewidth(.7)
    rng = np.random.default_rng(0)  # Horizontal display jitter only; no resampling.
    for position, values in zip(x, distribution):
        ax.scatter(position + rng.uniform(-.14, .14, len(values)), values,
                   s=5, color=color, alpha=.25, linewidths=0, zorder=2)
    ax.plot(x, [v.mean() for v in distribution], 'o-', color=color, ms=4,
            lw=1.2, label='Observed group mean', zorder=4)
