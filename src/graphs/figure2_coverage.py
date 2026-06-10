import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({

    # Figure size (single-column paper friendly)
    'figure.figsize': (7.0, 5.5),

    # Typography
    'font.size': 16,
    'axes.labelsize': 28,
    'axes.titlesize': 10,
    'legend.fontsize': 16,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,

    # Font
    'font.family': 'sans-serif',

    # Line settings
    'lines.linewidth': 1.8,

    # PDF export
#     'pdf.fonttype': 42,
#     'ps.fonttype': 42,

    # Layout
    'figure.constrained_layout.use': False
})

# ============================================================
# COMMON X-AXIS
# ============================================================

alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]



# ============================================================
# STANDARD PAPER MARGINS
# ============================================================

def apply_standard_margins():

    plt.subplots_adjust(
        left=0.16,
        right=0.98,
        top=0.96,
        bottom=0.18
    )

    plt.grid(
        True,
        linestyle='--',
        linewidth=0.6,
        alpha=0.30
    )

    plt.xlim(0.04, 0.31)

    # Ensure fixed alpha ticks
    plt.xticks(alphas)



# ============================================================
# GRAPH 1: EMPIRICAL RESPONSE COVERAGE
# ============================================================

def plot_response_coverage():

    su_cov = [0.810, 0.820, 0.835, 0.852, 0.865, 0.880]
    ddcrp_cov = [0.942, 0.884, 0.830, 0.791, 0.750, 0.710]
    cap_cov = [0.951, 0.902, 0.850, 0.814, 0.770, 0.730]
    acse_cov = [0.971, 0.923, 0.875, 0.835, 0.790, 0.750]

    plt.figure()

    plt.plot(
        alphas,
        su_cov,
        label='SU',
        color='gray',
        linestyle='--',
        marker='s',
        markersize=4.5
    )

    plt.plot(
        alphas,
        ddcrp_cov,
        label='DDCRP-CP',
        color='green',
        linestyle='-.',
        marker='^',
        markersize=4.5
    )

    plt.plot(
        alphas,
        cap_cov,
        label='CAP',
        color='purple',
        linestyle=':',
        marker='d',
        markersize=4.5
    )

    plt.plot(
        alphas,
        acse_cov,
        label='ACSE (Ours)',
        color='tab:blue',
        linestyle='-',
        marker='o',
        markersize=5.5,
        linewidth=2.2
    )

    plt.xlabel(r'Miscoverage Level ($\alpha$)')
    plt.ylabel('Emp. Response Cov.')

    plt.legend(
        loc='lower left',
        frameon=True,
        edgecolor='black'
    )

    apply_standard_margins()

    plt.savefig(os.path.join(root_path, 'EmpiricalResponseCoverage.pdf'),
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.02
    )

    plt.show()
    plt.close()



# ============================================================
# GRAPH 2: EMPIRICAL SELECTIVE RISK
# ============================================================

def plot_risk():

    acse_risk = np.array([0.044, 0.088, 0.132, 0.178, 0.222, 0.265])
    su_risk = np.array([0.064, 0.118, 0.172, 0.225, 0.282, 0.340])
    ddcrp_risk = np.array([0.038, 0.076, 0.115, 0.152, 0.190, 0.228])
    cap_risk = np.array([0.046, 0.093, 0.141, 0.188, 0.235, 0.282])

    plt.figure()

    # Guaranteed diagonal
    plt.plot(
        [0, 0.35],
        [0, 0.35],
        color='black',
        linestyle='--',
        alpha=0.8,
        linewidth=1.5,
        label='Guaranteed Risk'
    )

    plt.plot(
        alphas,
        su_risk,
        marker='s',
        color='#990000',
        linestyle='--',
        markersize=4.5,
        label='SU'
    )

    plt.plot(
        alphas,
        ddcrp_risk,
        marker='^',
        color='green',
        linestyle='-.',
        markersize=4.5,
        label='DDCRP-CP'
    )

    plt.plot(
        alphas,
        cap_risk,
        marker='d',
        color='purple',
        linestyle=':',
        markersize=4.5,
        label='CAP'
    )

    plt.plot(
        alphas,
        acse_risk,
        marker='o',
        color='tab:blue',
        linestyle='-',
        markersize=5.5,
        linewidth=2.2,
        label='ACSE (Ours)'
    )

    # Violation region
    plt.fill_between(
        [0, 0.35],
        [0, 0.35],
        0.4,
        color='#990000',
        alpha=0.05
    )

    plt.text(
        0.190,
        0.295,
        'Risk Violation\n      Zone',
        fontsize=17,
        color='#990000',
        alpha=0.7
    )

    plt.xlabel(r'Miscoverage Level ($\alpha$)')
    plt.ylabel('Emp. Selective Risk')

    plt.ylim(0.03, 0.35)

    plt.legend(
        loc='upper left',
        frameon=True,
        edgecolor='black'
    )

    apply_standard_margins()

    plt.savefig(os.path.join(root_path, 'SelectiveRisk.pdf'),
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.02
    )

    plt.show()
    plt.close()



# ============================================================
# GRAPH 3: EMPIRICAL PROMPT COVERAGE
# ============================================================

def plot_prompt_coverage():

    guaranteed_bound = [1 - a for a in alphas]

    acse_ctc = [0.971, 0.923, 0.885, 0.835, 0.782, 0.731]
    cap_ctc = [0.951, 0.902, 0.864, 0.814, 0.755, 0.702]
    ddcrp_ctc = [0.942, 0.884, 0.832, 0.791, 0.734, 0.685]
    su_ctc = [0.810, 0.820, 0.835, 0.852, 0.871, 0.892]

    plt.figure()

    plt.plot(
        alphas,
        guaranteed_bound,
        linestyle='--',
        color='black',
        linewidth=1.5,
        alpha=0.8,
        label='Guaranteed Bound'
    )

    plt.plot(
        alphas,
        su_ctc,
        marker='s',
        linestyle='--',
        color='gray',
        markersize=4.5,
        label='SU'
    )

    plt.plot(
        alphas,
        ddcrp_ctc,
        marker='^',
        linestyle='-.',
        color='green',
        markersize=4.5,
        label='DDCRP-CP'
    )

    plt.plot(
        alphas,
        cap_ctc,
        marker='d',
        linestyle=':',
        color='purple',
        markersize=4.5,
        label='CAP'
    )

    plt.plot(
        alphas,
        acse_ctc,
        marker='o',
        linestyle='-',
        color='tab:blue',
        linewidth=2.2,
        markersize=5.5,
        label='ACSE (Ours)'
    )

    plt.xlabel(r'Miscoverage Level ($\alpha$)')
    plt.ylabel('Emp. Prompt Cov.')

    plt.ylim(0.65, 1.0)

    plt.legend(
        loc='lower left',
        frameon=True,
        edgecolor='black'
    )

    apply_standard_margins()

    plt.savefig(os.path.join(root_path, 'EmpiricalPromptCoverage.pdf'),
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.02
    )

    plt.show()
    plt.close()



# ============================================================
# GRAPH 4: ACCEPTANCE RATE
# ============================================================

def plot_acceptance():

    alphas_orig = np.array([0.05, 0.10, 0.20, 0.30])

    se_orig = np.array([34.2, 52.1, 68.4, 76.5])
    ddcrp_orig = np.array([42.5, 62.8, 79.1, 85.3])
    cap_orig = np.array([46.1, 65.4, 82.5, 89.1])
    acse_orig = np.array([55.4, 75.8, 89.4, 94.4])

    se_acc = np.interp(alphas, alphas_orig, se_orig)
    ddcrp_acc = np.interp(alphas, alphas_orig, ddcrp_orig)
    cap_acc = np.interp(alphas, alphas_orig, cap_orig)
    acse_acc = np.interp(alphas, alphas_orig, acse_orig)

    plt.figure()

    plt.plot(
        alphas,
        se_acc,
        label='SU',
        marker='s',
        linestyle='--',
        color='gray',
        markersize=4.5
    )

    plt.plot(
        alphas,
        ddcrp_acc,
        label='DDCRP-CP',
        marker='^',
        linestyle='-.',
        color='green',
        markersize=4.5
    )

    plt.plot(
        alphas,
        cap_acc,
        label='CAP',
        marker='d',
        linestyle=':',
        color='purple',
        markersize=4.5
    )

    plt.plot(
        alphas,
        acse_acc,
        label='ACSE (Ours)',
        marker='o',
        linestyle='-',
        color='tab:blue',
        linewidth=2.2,
        markersize=5.5
    )

    plt.xlabel(r'Miscoverage Level ($\alpha$)')
    plt.ylabel('Acceptance Rate (%)')

    plt.yticks(np.arange(30, 101, 10))

    plt.legend(
        loc='lower right',
        frameon=True,
        edgecolor='black'
    )

    apply_standard_margins()

    plt.savefig(os.path.join(root_path, 'AcceptanceRate.pdf'),
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.02
    )

    plt.show()
    plt.close()



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    plot_response_coverage()
    plot_risk()
    plot_prompt_coverage()
    plot_acceptance()

    print(
        "All figures generated successfully."
    )
    
    
    