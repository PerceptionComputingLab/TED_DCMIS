import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = "times-new-roman.ttf"
font_prop = fm.FontProperties(fname=font_path, size=19)


def add_values_on_top_of_bars(ax, data, bar_position, shift=0.005, offset=[]):
    """
    Add values on top of the bars in a bar chart.

    Parameters:
    - ax: The axis on which bars are plotted.
    - data: The data values to be added on top.
    - bar_positions: The x-axis positions of the bars.
    - bar_width: Width of the bars.
    - shift: Vertical shift for the text for better visualization.
    - offset: Additional vertical offset for the text.
    """
    for i, value in enumerate(data):
        ax.text(
            bar_positions[i],
            value + shift + offset[i] / 100.0,
            round(value, 3),
            ha="center",
            va="bottom",
            fontsize=10,
        )


# Creating data
prostate = ["RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK"]
offsets = [0.01, 0.02, 0.03, 0.04]
seq = np.array(
    [
        0.38413072097203776,
        0.4006589883840904,
        0.4972383785915587,
        0.38588250238986715,
        0.5016571520403096,
        0.5407959914510456,
    ]
)
bst = np.array(
    [
        0.45027472056881923,
        0.42915558572119966,
        0.5036675793635842,
        0.44268759668510543,
        0.5125576271031702,
        0.49315115895113576,
    ]
)
kd = np.array(
    [
        0.32448884265258393,
        0.357195225833993,
        0.46751843421539085,
        0.39263390044949403,
        0.3645766436321109,
        0.40715944010097743,
    ]
)
mas = np.array(
    [
        0.17628126541251454,
        0.2999555688816964,
        0.3721556845074497,
        0.3116487403591346,
        0.3663483144470182,
        0.18687656238565886,
    ]
)
plop = np.array(
    [
        0.17149389491680417,
        0.28008838773710354,
        0.37660046467268504,
        0.31423761617641327,
        0.279858345548347,
        0.23724976540407225,
    ]
)

# Plot
plt.figure(figsize=[8, 5])

# Use bar function from pyplot to create bar plots
bar_positions = np.arange(len(prostate))
# plt.bar(np.arange(len(prostate)) - 0.2, seq, width=0.1, color='royalblue', align='center', label='seq')
plt.bar(
    np.arange(len(prostate)) - 0.1,
    mas,
    width=0.1,
    color="darkgray",
    align="center",
    label="MAS",
)
add_values_on_top_of_bars(plt, mas, np.arange(len(prostate)) - 0.1, 0.1, offset=mas)
plt.bar(
    np.arange(len(prostate)),
    kd,
    width=0.1,
    color="firebrick",
    align="center",
    label="KD",
)
add_values_on_top_of_bars(plt, kd, np.arange(len(prostate)), 0.1, offset=kd)
plt.bar(
    np.arange(len(prostate)) + 0.1,
    plop,
    width=0.1,
    color="seagreen",
    align="center",
    label="PLOP",
)
add_values_on_top_of_bars(plt, plop, np.arange(len(prostate)) + 0.1, 0.1, offset=plop)
plt.bar(
    np.arange(len(prostate)) + 0.2,
    bst,
    width=0.1,
    color="darkorange",
    align="center",
    label="AKT",
)
add_values_on_top_of_bars(plt, bst, np.arange(len(prostate)) + 0.2, 0.1, offset=bst)

plt.ylabel("Uncertainty on misclassified regions", fontproperties=font_prop)
plt.ylim(0.1, 0.6)
plt.xticks(np.arange(len(prostate)), prostate, fontproperties=font_prop)
plt.title("Prostate", fontproperties=font_prop)
plt.legend(ncol=len(prostate), loc="upper center", prop=font_prop)
plt.tight_layout()
plt.savefig("backgroundshift_prostate.png", dpi=300)
plt.show()

# Creating data
cardiac_i = ["Siemens", "Philips", "GE", "Canon"]
seq_i = np.array(
    [0.5760730471390291, 0.5842209353680831, 0.6373696909534111, 0.6527687312844521]
)
mas_i = np.array(
    [0.25301220896731097, 0.26642325367403247, 0.31340247622198636, 0.3227312696264851]
)
kd_i = np.array(
    [0.3883382876644014, 0.4114479870081355, 0.4867034385985664, 0.4634157418586244]
)
plop_i = np.array(
    [0.5118577664774409, 0.5198315103637569, 0.589350139238527, 0.5960363550752685]
)
bst_i = np.array(
    [0.5217978568437198, 0.5513720186764558, 0.6086049904938543, 0.5610067867309436]
)

cardiac_o = ["Siemens", "Philips", "GE", "Canon"]
seq_o = np.array(
    [0.44292827268449264, 0.4539756019800355, 0.4818948037037849, 0.4735828782778371]
)
mas_o = np.array(
    [0.5950867197812835, 0.6179007992526409, 0.6348829639835377, 0.6098119841200924]
)
kd_o = np.array(
    [0.6215375353008024, 0.6212542319708404, 0.6624320063680352, 0.6274086879854167]
)
plop_o = np.array(
    [0.5388622997012137, 0.5617190445404212, 0.6148527685753614, 0.5683402070293917]
)
bst_o = np.array(
    [0.6368418745846033, 0.6431061970843159, 0.6865985209028855, 0.6363563235911008]
)

cardiac_r = ["Siemens", "Philips", "GE", "Canon"]
seq_r = np.array(
    [0.4907239652518729, 0.533277628965283, 0.47660760555043963, 0.5153066662986938]
)
mas_r = np.array(
    [0.6013288058358217, 0.642397384142038, 0.6462454301268689, 0.6324799002935746]
)
kd_r = np.array(
    [0.474258719705713, 0.5471400901365141, 0.5326042587843935, 0.5605235832594619]
)
plop_r = np.array(
    [0.5446188450211803, 0.5989395012576127, 0.55961064535226, 0.604804952247668]
)
bst_r = np.array(
    [0.5911634932325895, 0.6395962415787626, 0.6481961155662849, 0.6426226344626859]
)

# Create subplot
fig, axs = plt.subplots(1, 3, figsize=[16, 5])

# Use bar function from pyplot to create bar plots
# axs[0].bar(np.arange(len(cardiac_i)) - 0.2, seq_i, width=0.1, color='royalblue', align='center', label='seq')
axs[0].bar(
    np.arange(len(cardiac_i)) - 0.1,
    mas_i,
    width=0.1,
    color="darkgray",
    align="center",
    label="MAS",
)
add_values_on_top_of_bars(
    axs[0], mas_i, np.arange(len(cardiac_i)) - 0.1, 0.1, offset=mas_i
)
axs[0].bar(
    np.arange(len(cardiac_i)),
    kd_i,
    width=0.1,
    color="firebrick",
    align="center",
    label="KD",
)
add_values_on_top_of_bars(axs[0], kd_i, np.arange(len(cardiac_i)), 0.1, offset=kd_i)
axs[0].bar(
    np.arange(len(cardiac_i)) + 0.1,
    plop_i,
    width=0.1,
    color="seagreen",
    align="center",
    label="PLOP",
)
add_values_on_top_of_bars(
    axs[0], plop_i, np.arange(len(cardiac_i)) + 0.1, 0.1, offset=plop_i
)
axs[0].bar(
    np.arange(len(cardiac_i)) + 0.2,
    bst_i,
    width=0.1,
    color="darkorange",
    align="center",
    label="AKT",
)
add_values_on_top_of_bars(
    axs[0], bst_i, np.arange(len(cardiac_i)) + 0.2, 0.1, offset=bst_i
)
axs[0].set_ylabel("Uncertainty on misclassified regions", fontproperties=font_prop)
axs[0].set_ylim(0.2, 0.7)
axs[0].set_title("LV-endo", fontproperties=font_prop)
axs[0].set_xticks(np.arange(len(cardiac_i)))
axs[0].set_xticklabels(cardiac_i, fontproperties=font_prop)

# axs[1].bar(np.arange(len(cardiac_o)) - 0.2, seq_o, width=0.1, color='royalblue', align='center', label='seq')
axs[1].bar(
    np.arange(len(cardiac_o)) - 0.1,
    mas_o,
    width=0.1,
    color="darkgray",
    align="center",
    label="MAS",
)
add_values_on_top_of_bars(
    axs[1], mas_o, np.arange(len(cardiac_o)) - 0.1, 0.1, offset=mas_o
)
axs[1].bar(
    np.arange(len(cardiac_o)),
    kd_o,
    width=0.1,
    color="firebrick",
    align="center",
    label="KD",
)
add_values_on_top_of_bars(axs[1], kd_o, np.arange(len(cardiac_o)), 0.1, offset=kd_o)
axs[1].bar(
    np.arange(len(cardiac_o)) + 0.1,
    plop_o,
    width=0.1,
    color="seagreen",
    align="center",
    label="PLOP",
)
add_values_on_top_of_bars(
    axs[1], plop_o, np.arange(len(cardiac_o)) + 0.1, 0.1, offset=plop_o
)
axs[1].bar(
    np.arange(len(cardiac_o)) + 0.2,
    bst_o,
    width=0.1,
    color="darkorange",
    align="center",
    label="AKT",
)
add_values_on_top_of_bars(
    axs[1], bst_o, np.arange(len(cardiac_o)) + 0.2, 0.1, offset=bst_o
)
# axs[1].set_ylabel('Uncertainty')
axs[1].set_ylim(0.2, 0.7)
axs[1].set_title("LV-epi", fontproperties=font_prop)
axs[1].set_xticks(np.arange(len(cardiac_o)))
axs[1].set_xticklabels(cardiac_o, fontproperties=font_prop)

# axs[2].bar(np.arange(len(cardiac_r)) - 0.2, seq_r, width=0.1, color='royalblue', align='center', label='seq')
axs[2].bar(
    np.arange(len(cardiac_r)) - 0.1,
    mas_r,
    width=0.1,
    color="darkgray",
    align="center",
    label="MAS",
)
add_values_on_top_of_bars(
    axs[2], mas_r, np.arange(len(cardiac_r)) - 0.1, 0.1, offset=mas_r
)
axs[2].bar(
    np.arange(len(cardiac_r)),
    kd_r,
    width=0.1,
    color="firebrick",
    align="center",
    label="KD",
)
add_values_on_top_of_bars(axs[2], kd_r, np.arange(len(cardiac_r)), 0.1, offset=kd_r)
axs[2].bar(
    np.arange(len(cardiac_r)) + 0.1,
    plop_r,
    width=0.1,
    color="seagreen",
    align="center",
    label="PLOP",
)
add_values_on_top_of_bars(
    axs[2], plop_r, np.arange(len(cardiac_r)) + 0.1, 0.1, offset=plop_r
)
axs[2].bar(
    np.arange(len(cardiac_r)) + 0.2,
    bst_r,
    width=0.1,
    color="darkorange",
    align="center",
    label="AKT",
)
add_values_on_top_of_bars(
    axs[2], bst_r, np.arange(len(cardiac_r)) + 0.2, 0.1, offset=bst_r
)
# axs[2].set_ylabel('Uncertainty')
axs[2].set_ylim(0.2, 0.7)
axs[2].set_title("RV", fontproperties=font_prop)
axs[2].set_xticks(np.arange(len(cardiac_r)))
axs[2].set_xticklabels(cardiac_r, fontproperties=font_prop)

handles, labels = axs[0].get_legend_handles_labels()

plt.tight_layout()
plt.savefig("background_cardiac.png", dpi=300)
plt.show()

legend_fig = plt.figure(figsize=(7, 1))
plt.legend(handles, labels, loc="center", ncol=4, prop=font_prop)
plt.axis("off")
plt.tight_layout()
legend_fig.savefig("legend.png", dpi=300)

# Plotting prostate and cardiac data together in a single figure

fig, axs = plt.subplots(1, 4, figsize=(24, 5))


def add_values_on_top_of_bars(
    ax, data, bar_positions, bar_width, shift=0.005, offset=0.0
):
    """
    Add values on top of the bars in a bar chart.

    Parameters:
    - ax: The axis on which bars are plotted.
    - data: The data values to be added on top.
    - bar_positions: The x-axis positions of the bars.
    - bar_width: Width of the bars.
    - shift: Vertical shift for the text for better visualization.
    - offset: Additional vertical offset for the text.
    """
    for i, value in enumerate(data):
        ax.text(
            bar_positions[i],
            value + shift + offset,
            round(value, 3),
            ha="center",
            va="bottom",
            fontsize=10,
        )


# Plot for prostate
bar_positions = np.arange(len(prostate))
bar_width = 0.2
offsets = [0.01, 0.02, 0.03, 0.04]
axs[0].bar(
    bar_positions - 0.1,
    mas,
    width=bar_width,
    color="darkgray",
    align="center",
    label="MAS",
)
add_values_on_top_of_bars(
    axs[0], mas, bar_positions - 0.1, bar_width, offset=offsets[0]
)

axs[0].bar(
    bar_positions, kd, width=bar_width, color="firebrick", align="center", label="KD"
)
add_values_on_top_of_bars(axs[0], kd, bar_positions, bar_width, offset=offsets[1])

axs[0].bar(
    bar_positions + 0.1,
    plop,
    width=bar_width,
    color="seagreen",
    align="center",
    label="PLOP",
)
add_values_on_top_of_bars(
    axs[0], plop, bar_positions + 0.1, bar_width, offset=offsets[2]
)

axs[0].bar(
    bar_positions + 0.2,
    bst,
    width=bar_width,
    color="darkorange",
    align="center",
    label="BST",
)
add_values_on_top_of_bars(
    axs[0], bst, bar_positions + 0.2, bar_width, offset=offsets[3]
)

axs[0].set_ylim(0.1, 0.6)
axs[0].set_title("Prostate", fontproperties=font_prop)
axs[0].set_xticks(bar_positions)
axs[0].set_xticklabels(prostate, fontproperties=font_prop)

# Plot for cardiac_i
axs[1].bar(
    np.arange(len(cardiac_i)) - 0.1,
    mas_i,
    width=bar_width,
    color="darkgray",
    align="center",
    label="MAS",
)
add_values_on_top_of_bars(
    axs[1], mas_i, np.arange(len(cardiac_i)) - 0.1, bar_width, offset=offsets[0]
)

axs[1].bar(
    np.arange(len(cardiac_i)),
    kd_i,
    width=bar_width,
    color="firebrick",
    align="center",
    label="KD",
)
add_values_on_top_of_bars(
    axs[1], kd_i, np.arange(len(cardiac_i)), bar_width, offset=offsets[1]
)

axs[1].bar(
    np.arange(len(cardiac_i)) + 0.1,
    plop_i,
    width=bar_width,
    color="seagreen",
    align="center",
    label="PLOP",
)
add_values_on_top_of_bars(
    axs[1], plop_i, np.arange(len(cardiac_i)) + 0.1, bar_width, offset=offsets[2]
)

axs[1].bar(
    np.arange(len(cardiac_i)) + 0.2,
    bst_i,
    width=bar_width,
    color="darkorange",
    align="center",
    label="BST",
)
add_values_on_top_of_bars(
    axs[1], bst_i, np.arange(len(cardiac_i)) + 0.2, bar_width, offset=offsets[3]
)

axs[1].set_ylim(0.2, 0.8)
axs[1].set_title("LV-endo", fontproperties=font_prop)
axs[1].set_xticks(np.arange(len(cardiac_i)))
axs[1].set_xticklabels(cardiac_i, fontproperties=font_prop)

# Plot for cardiac_o
axs[2].bar(
    np.arange(len(cardiac_o)) - 0.1,
    mas_o,
    width=bar_width,
    color="darkgray",
    align="center",
    label="MAS",
)
add_values_on_top_of_bars(
    axs[2], mas_o, np.arange(len(cardiac_o)) - 0.1, bar_width, offset=offsets[0]
)

axs[2].bar(
    np.arange(len(cardiac_o)),
    kd_o,
    width=bar_width,
    color="firebrick",
    align="center",
    label="KD",
)
add_values_on_top_of_bars(
    axs[2], kd_o, np.arange(len(cardiac_o)), bar_width, offset=offsets[1]
)

axs[2].bar(
    np.arange(len(cardiac_o)) + 0.1,
    plop_o,
    width=bar_width,
    color="seagreen",
    align="center",
    label="PLOP",
)
add_values_on_top_of_bars(
    axs[2], plop_o, np.arange(len(cardiac_o)) + 0.1, bar_width, offset=offsets[2]
)

axs[2].bar(
    np.arange(len(cardiac_o)) + 0.2,
    bst_o,
    width=bar_width,
    color="darkorange",
    align="center",
    label="BST",
)
add_values_on_top_of_bars(
    axs[2], bst_o, np.arange(len(cardiac_o)) + 0.2, bar_width, offset=offsets[3]
)

axs[2].set_ylim(0.2, 0.8)
axs[2].set_title("LV-epi", fontproperties=font_prop)
axs[2].set_xticks(np.arange(len(cardiac_o)))
axs[2].set_xticklabels(cardiac_o, fontproperties=font_prop)

# Plot for cardiac_r
axs[3].bar(
    np.arange(len(cardiac_r)) - 0.1,
    mas_r,
    width=bar_width,
    color="darkgray",
    align="center",
    label="MAS",
)
add_values_on_top_of_bars(
    axs[3], mas_r, np.arange(len(cardiac_r)) - 0.1, bar_width, offset=offsets[0]
)

axs[3].bar(
    np.arange(len(cardiac_r)),
    kd_r,
    width=bar_width,
    color="firebrick",
    align="center",
    label="KD",
)
add_values_on_top_of_bars(
    axs[3], kd_r, np.arange(len(cardiac_r)), bar_width, offset=offsets[1]
)

axs[3].bar(
    np.arange(len(cardiac_r)) + 0.1,
    plop_r,
    width=bar_width,
    color="seagreen",
    align="center",
    label="PLOP",
)
add_values_on_top_of_bars(
    axs[3], plop_r, np.arange(len(cardiac_r)) + 0.1, bar_width, offset=offsets[2]
)

axs[3].bar(
    np.arange(len(cardiac_r)) + 0.2,
    bst_r,
    width=bar_width,
    color="darkorange",
    align="center",
    label="BST",
)
add_values_on_top_of_bars(
    axs[3], bst_r, np.arange(len(cardiac_r)) + 0.2, bar_width, offset=offsets[3]
)

axs[3].set_ylim(0.2, 0.8)
axs[3].set_title("RV", fontproperties=font_prop)
axs[3].set_xticks(np.arange(len(cardiac_r)))
axs[3].set_xticklabels(cardiac_r, fontproperties=font_prop)

plt.tight_layout()
plt.show()
