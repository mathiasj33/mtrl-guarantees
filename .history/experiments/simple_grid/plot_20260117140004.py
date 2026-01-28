# ... keep everything above unchanged ...

    # ===== Left plot =====
    ax0 = axes[0]
    apply_anthropic_style(ax0)
    ax0.plot(
        xs,
        pdf,
        color=ANTHROPIC["red_dark"],
        linewidth=2.2,          # thinner
        solid_capstyle="round",
    )
    ax0.set_title(r"\textbf{Slip Probability Density Function}", fontsize=13, color=ANTHROPIC["ink"], pad=8)
    ax0.set_xlabel(r"\textbf{Slip Probability}", fontsize=14, color=ANTHROPIC["ink"], labelpad=6)
    ax0.set_ylabel(r"\textbf{Density}", fontsize=14, color=ANTHROPIC["ink"], labelpad=6)

    # ===== Right plot =====
    # computed = dark; actual = light/high-contrast
    ax1.plot(
        guarantees["guarantees"],
        guarantees["probs"],
        color=ANTHROPIC["red_dark"],
        linewidth=2.2,          # thinner
        solid_capstyle="round",
        label=r"\textbf{Computed Guarantees}",
    )
    ax1.plot(
        actual_guarantees["guarantees"],
        actual_guarantees["probs"],
        color=ANTHROPIC["orange_light"],
        linewidth=2.4,          # thinner (still a touch thicker so it reads)
        linestyle="--",
        dash_capstyle="round",
        label=r"\textbf{Actual Guarantees}",
    )

# ... keep everything below unchanged ...