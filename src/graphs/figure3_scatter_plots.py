# -*- coding: utf-8 -*-

plt.rcParams.update({

    # Figure size (single-column paper friendly)
#     'figure.figsize': (7.0, 5.5),

    # Typography
#     'font.size': 16,
#     'axes.labelsize': 28,
#     'axes.titlesize': 10,
#     'legend.fontsize': 16,
#     'xtick.labelsize': 14,
#     'ytick.labelsize': 14,

    # Font
    'font.family': 'sans-serif',

    # Line settings
#     'lines.linewidth': 1.8,

    # PDF export
#     'pdf.fonttype': 42,
#     'ps.fonttype': 42,

    # Layout
#     'figure.constrained_layout.use': False
})



### TE Scores

import json
import matplotlib.pyplot as plt

n_correct = 680

with open(os.path.join(root_path,"acse_scores_1000.json"), "r", encoding="utf-8") as f:
    acse_scores = json.load(f)

with open(os.path.join(root_path,"te_scores_1000.json"), "r", encoding="utf-8") as f:
    te_scores = json.load(f)

prompt_ids = sorted(acse_scores.keys())

x_te = []
y_acse = []
colors = []

for prompt_id in prompt_ids:
    idx = int(prompt_id.split("_")[1])

    x_te.append(te_scores[prompt_id])
    y_acse.append(acse_scores[prompt_id])

    if idx <= n_correct:
        colors.append("#2f5f8f")
    else:
        colors.append("#b44a4a")

plt.figure(figsize=(7.2, 5.5))

plt.scatter(
    x_te,
    y_acse,
    c=colors,
    s=34,
    alpha=0.62,
    edgecolors="none"
)

plt.axhline(0.48, color="black", linestyle=":", linewidth=3, alpha=0.7)
plt.axvline(0.70, color="black", linestyle=":", linewidth=3, alpha=0.7)

plt.xlabel("TE Confidence", fontsize=38)
plt.ylabel(r"ACSE Uncertainty", fontsize=35) # ($\hat{u}$)

plt.xlim(0.4, 1.01)
plt.ylim(-0.05, 1.0)

plt.xticks([0.4, 0.6, 0.8, 1.0], fontsize=22)
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=22)
plt.tick_params(axis="both", labelsize=20)

plt.scatter([], [], c="#2f5f8f", s=60, alpha=0.7, label="Correct")
plt.scatter([], [], c="#b44a4a", s=60, alpha=0.7, label="Incorrect")
# plt.legend(loc="upper left", fontsize=22, frameon=True)

plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(root_path, 'ACSE_TE.pdf'),
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.02
    )
plt.show()
plt.close()


"""### P(True) Scores"""

import json
import matplotlib.pyplot as plt

n_correct = 680

with open(os.path.join(root_path,"acse_scores_1000.json"), "r", encoding="utf-8") as f:
    acse_scores = json.load(f)

with open(os.path.join(root_path,"ptrue_scores_1000.json"), "r", encoding="utf-8") as f:
    ptrue_scores = json.load(f)

prompt_ids = sorted(acse_scores.keys())

x_ptrue = []
y_acse = []
colors = []

for prompt_id in prompt_ids:
    idx = int(prompt_id.split("_")[1])

    x_ptrue.append(ptrue_scores[prompt_id])
    y_acse.append(acse_scores[prompt_id])

    if idx <= n_correct:
        colors.append("#2f5f8f")
    else:
        colors.append("#b44a4a")

plt.figure(figsize=(6.3, 5.5))

plt.scatter(
    x_ptrue,
    y_acse,
    c=colors,
    s=34,
    alpha=0.62,
    edgecolors="none"
)

plt.axhline(0.48, color="black", linestyle=":", linewidth=3, alpha=0.7)
plt.axvline(0.70, color="black", linestyle=":", linewidth=3, alpha=0.7)

plt.xlabel("P(True) Confidence", fontsize=38)
# plt.ylabel(r"ACSE Uncertainty ($\hat{u}$)", fontsize=28)

plt.xlim(0.4, 1.01)
# plt.ylim(-0.05, 1.0)

plt.xticks([0.4, 0.6, 0.8, 1.0], fontsize=22)
# plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
plt.tick_params(axis="y", left=False, labelleft=False)
plt.tick_params(axis="x", labelsize=20)

plt.scatter([], [], c="#2f5f8f", s=60, alpha=0.7, label="Correct")
plt.scatter([], [], c="#b44a4a", s=60, alpha=0.7, label="Incorrect")
# plt.legend(loc="upper left", fontsize=22, frameon=True)

plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(root_path, 'ACSE_PTRUE.pdf'),
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.02
    )
plt.show()
plt.close()


"""### EigV Scores"""

import json
import matplotlib.pyplot as plt

n_correct = 680

with open(os.path.join(root_path,"acse_scores_1000.json"), "r", encoding="utf-8") as f:
    acse_scores = json.load(f)

with open(os.path.join(root_path,"eigv_scores_1000.json"), "r", encoding="utf-8") as f:
    eigv_scores = json.load(f)

prompt_ids = sorted(acse_scores.keys())

x_eigv = []
y_acse = []
colors = []

for prompt_id in prompt_ids:
    idx = int(prompt_id.split("_")[1])

    x_eigv.append(eigv_scores[prompt_id])
    y_acse.append(acse_scores[prompt_id])

    if idx <= n_correct:
        colors.append("#2f5f8f")
    else:
        colors.append("#b44a4a")

plt.figure(figsize=(7.2, 5.5))

plt.scatter(
    x_eigv,
    y_acse,
    c=colors,
    s=34,
    alpha=0.62,
    edgecolors="none"
)

plt.axhline(0.48, color="black", linestyle=":", linewidth=3, alpha=0.7)
plt.axvline(0.70, color="black", linestyle=":", linewidth=3, alpha=0.7)

plt.xlabel("EigV Confidence", fontsize=38)
plt.ylabel(r"ACSE Uncertainty", fontsize=35) # ($\hat{u}$)

plt.xlim(0.4, 1.01)
plt.ylim(-0.05, 1.0)

plt.xticks([0.4, 0.6, 0.8, 1.0], fontsize=22)
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=22)
plt.tick_params(axis="both", labelsize=20)

plt.scatter([], [], c="#2f5f8f", s=60, alpha=0.7, label="Correct")
plt.scatter([], [], c="#b44a4a", s=60, alpha=0.7, label="Incorrect")
# plt.legend(loc="upper left", fontsize=22, frameon=True)

plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(root_path, 'ACSE_EIGV.pdf'),
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.02
    )
plt.show()
plt.close()


"""### SU"""

import json
import matplotlib.pyplot as plt

n_correct = 680

with open(os.path.join(root_path,"acse_scores_1000.json"), "r", encoding="utf-8") as f:
    acse_scores = json.load(f)

with open(os.path.join(root_path,"su_scores_1000.json"), "r", encoding="utf-8") as f:
    su_scores = json.load(f)

prompt_ids = sorted(acse_scores.keys())

x_su = []
y_acse = []
colors = []

for prompt_id in prompt_ids:
    idx = int(prompt_id.split("_")[1])

    x_su.append(su_scores[prompt_id])
    y_acse.append(acse_scores[prompt_id])

    if idx <= n_correct:
        colors.append("#2f5f8f")
    else:
        colors.append("#b44a4a")

plt.figure(figsize=(6.3, 5.5))

plt.scatter(
    x_su,
    y_acse,
    c=colors,
    s=34,
    alpha=0.62,
    edgecolors="none"
)

plt.axhline(0.48, color="black", linestyle=":", linewidth=3, alpha=0.7)
plt.axvline(0.70, color="black", linestyle=":", linewidth=3, alpha=0.7)

plt.xlabel("SU Confidence", fontsize=38)
# plt.ylabel(r"ACSE Uncertainty ($\hat{u}$)", fontsize=28)

plt.xlim(0.4, 1.01)
# plt.ylim(-0.05, 1.0)

plt.xticks([0.4, 0.6, 0.8, 1.0], fontsize=22)
# plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=22)
plt.tick_params(axis="y", left=False, labelleft=False)
plt.tick_params(axis="x", labelsize=20)

plt.scatter([], [], c="#2f5f8f", s=60, alpha=0.7, label="Correct")
plt.scatter([], [], c="#b44a4a", s=60, alpha=0.7, label="Incorrect")
# plt.legend(loc="upper left", fontsize=22, frameon=True)

plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(root_path, 'ACSE_SU.pdf'),
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.02
    )
plt.show()
plt.close()


"""### CAP"""

import json
import matplotlib.pyplot as plt

n_correct = 680

with open(os.path.join(root_path,"acse_scores_1000.json"), "r", encoding="utf-8") as f:
    acse_scores = json.load(f)

with open(os.path.join(root_path,"cap_scores_1000.json"), "r", encoding="utf-8") as f:
    cap_scores = json.load(f)

prompt_ids = sorted(acse_scores.keys())

x_cap = []
y_acse = []
colors = []

for prompt_id in prompt_ids:
    idx = int(prompt_id.split("_")[1])

    x_cap.append(cap_scores[prompt_id])
    y_acse.append(acse_scores[prompt_id])

    if idx <= n_correct:
        colors.append("#2f5f8f")
    else:
        colors.append("#b44a4a")

plt.figure(figsize=(6.3, 5.5))

plt.scatter(
    x_cap,
    y_acse,
    c=colors,
    s=34,
    alpha=0.62,
    edgecolors="none"
)

plt.axhline(0.48, color="black", linestyle=":", linewidth=3, alpha=0.7)
plt.axvline(0.70, color="black", linestyle=":", linewidth=3, alpha=0.7)

plt.xlabel("CAP Confidence", fontsize=38)
# plt.ylabel(r"ACSE Uncertainty ($\hat{u}$)", fontsize=28)

plt.xlim(0.4, 1.01)
# plt.ylim(-0.05, 1.0)

plt.xticks([0.4, 0.6, 0.8, 1.0], fontsize=22)
# plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=22)
plt.tick_params(axis="y", left=False, labelleft=False)
plt.tick_params(axis="x", labelsize=20)

plt.scatter([], [], c="#2f5f8f", s=60, alpha=0.7, label="Correct")
plt.scatter([], [], c="#b44a4a", s=60, alpha=0.7, label="Incorrect")
# plt.legend(loc="upper left", fontsize=22, frameon=True)

plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(root_path, 'ACSE_CAP.pdf'),
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.02
    )
plt.show()
plt.close()


"""### DDRCP-CP"""

import json
import matplotlib.pyplot as plt

n_correct = 680

with open(os.path.join(root_path,"acse_scores_1000.json"), "r", encoding="utf-8") as f:
    acse_scores = json.load(f)

with open(os.path.join(root_path,"ddcrp_cp_scores_1000.json"), "r", encoding="utf-8") as f:
    ddcrp_cp_scores = json.load(f)

prompt_ids = sorted(acse_scores.keys())

x_ddcrp = []
y_acse = []
colors = []

for prompt_id in prompt_ids:
    idx = int(prompt_id.split("_")[1])

    x_ddcrp.append(ddcrp_cp_scores[prompt_id])
    y_acse.append(acse_scores[prompt_id])

    if idx <= n_correct:
        colors.append("#2f5f8f")
    else:
        colors.append("#b44a4a")

plt.figure(figsize=(6.3, 5.5))

plt.scatter(
    x_ddcrp,
    y_acse,
    c=colors,
    s=34,
    alpha=0.62,
    edgecolors="none"
)

plt.axhline(0.48, color="black", linestyle=":", linewidth=3, alpha=0.7)
plt.axvline(0.70, color="black", linestyle=":", linewidth=3, alpha=0.7)

plt.xlabel("DDCRP-CP Confidence", fontsize=38)
# plt.ylabel(r"ACSE Uncertainty ($\hat{u}$)", fontsize=28)

plt.xlim(0.4, 1.01)
# plt.ylim(-0.05, 1.0)

plt.xticks([0.4, 0.6, 0.8, 1.0], fontsize=22)
# plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=22)
plt.tick_params(axis="y", left=False, labelleft=False)
plt.tick_params(axis="x", labelsize=20)

plt.scatter([], [], c="#2f5f8f", s=60, alpha=0.7, label="Correct")
plt.scatter([], [], c="#b44a4a", s=60, alpha=0.7, label="Incorrect")
# plt.legend(loc="upper left", fontsize=22, frameon=True)

plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(root_path, 'ACSE_DDCRP-CP.pdf'),
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.02
    )
plt.show()
plt.close()


