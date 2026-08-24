"""Optional matplotlib renderer for CLI / local backtest runs."""

from __future__ import annotations

from algo_trading_engine.plotting.spec import PlotSpec


def show_plot(spec: PlotSpec) -> None:
    """Render a PlotSpec with matplotlib and display it."""
    # Deferred: matplotlib is an optional [plot] extra.
    import matplotlib.dates as mdates  # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    frame = spec.frame
    timestamps = frame["timestamp"]

    fig, ax1 = plt.subplots(figsize=(14, 7))
    left_cols = [
        col
        for col in frame.columns
        if col not in {"timestamp", *spec.right_axis}
    ]
    for col in left_cols:
        ax1.plot(timestamps, frame[col], label=col, linewidth=1.5)

    for key, marker_frame in spec.markers.items():
        if marker_frame.empty:
            continue
        value_col = next(
            (c for c in marker_frame.columns if c != "timestamp"),
            None,
        )
        if value_col is None:
            continue
        ax1.scatter(
            marker_frame["timestamp"],
            marker_frame[value_col],
            label=key,
            s=60,
            zorder=5,
        )

    if spec.y_label:
        ax1.set_ylabel(spec.y_label)
    ax1.grid(True, alpha=0.3)

    if spec.right_axis:
        ax2 = ax1.twinx()
        for col in spec.right_axis:
            if col in frame.columns:
                ax2.plot(timestamps, frame[col], label=col, linestyle="--", alpha=0.7)
        ax2.set_ylabel(", ".join(spec.right_axis))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    else:
        ax1.legend(loc="upper left")

    if spec.title:
        ax1.set_title(spec.title, fontsize=14, fontweight="bold")

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    backend = plt.get_backend().lower()
    if backend in {"agg", "svg", "pdf", "ps", "cairo", "template"}:
        plt.close(fig)
        return
    plt.show()
    plt.close(fig)
