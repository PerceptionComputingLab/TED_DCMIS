import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import font_manager as fm

'''
    dataset: prostate
    approach: kd, kd_shift, kd_ours, kd_ours_wn, kd_shift_wn
    approach: kd, kd + akt + gvu, kd + akt + gvu + ska, kd + akt + ska, kd + akt
    data source: analysis/table_figure.py
'''

font_path = 'times-new-roman.ttf'
font_prop = fm.FontProperties(fname=font_path, size=19)
font_prop_small = fm.FontProperties(fname=font_path, size=10)
kd = [0.8617935643754502, 0.7758046870384094, 0.6919511633751515, 0.7910741134312795, 0.7649577736170894,
      0.709054743972823]
kd_shift = [0.8617935643754502, 0.808504617938013, 0.7463886077992193, 0.7906880830345018, 0.7736797186903139,
            0.753276296328190]
kd_ours = [0.8617935643754502, 0.8024335712835421, 0.7814310897196098, 0.7979569479941697, 0.7741190024182889,
           0.7592365099398797]
kd_ours_wn = [0.8617935643754502, 0.7896470582689188, 0.7833870749773011, 0.7938973844478853, 0.7678110631134125,
              0.7441583363816765]
kd_shift_wn = [0.8617935643754502, 0.7781642783033855, 0.7280486381588419, 0.779117288720355, 0.7665519146340826,
               0.7409179804042783]

kd_std = np.std(kd)
kd_shift_std = np.std(kd_shift)
kd_ours_std = np.std(kd_ours)
kd_ours_wn_std = np.std(kd_ours_wn)
kd_shift_wn_std = np.std(kd_shift_wn)

x = ['RUNMC', 'BMC', 'I2CVB', 'UCL', 'BIDMC', 'HK']

fig, axs = plt.subplots(2, 1, figsize=(6, 5))
axs[0].plot(kd, label='KD', marker='o', linestyle='-', color='darkblue')
axs[0].plot(kd_shift, label='TED - GVU', marker='o', linestyle='-', color='darkorange')
axs[0].plot(kd_ours, label='TED (Ours)', marker='o', linestyle='-', color='darkgreen')
axs[0].fill_between(x, np.array(kd_ours) - kd_ours_std, np.array(kd_ours) + kd_ours_std, alpha=0.2, color='darkgreen')
axs[0].set_xticks([])
axs[0].set_ylabel('AVG DICE', fontproperties=font_prop)
axs[0].legend(loc='lower left', prop=font_prop_small)

axs[1].plot(kd, label='KD', marker='o', linestyle='-', color='darkblue')
axs[1].plot(kd_shift_wn, label='KD + AKT', marker='o', linestyle='-', color='darkred')
axs[1].plot(kd_ours_wn, label='KD + AKT + GVU', marker='o', linestyle='-', color='darkviolet')
axs[1].fill_between(x, np.array(kd_ours_wn) - kd_ours_wn_std, np.array(kd_ours_wn) + kd_ours_wn_std, alpha=0.2,
                    color='darkviolet')
x = range(0, 6, 1)
plt.xticks(x, ('RUNMC', 'BMC', 'I2CVB', 'UCL', 'BIDMC', 'HK',), fontproperties=font_prop)
axs[1].set_ylabel('AVG DICE', fontproperties=font_prop)
axs[1].legend(loc='lower left', prop=font_prop_small)

plt.tight_layout()
plt.savefig('prostate_dice.png')
plt.show()

kd = [14.294088756514212, 29.92599971179357, 35.79183013191914, 13.38655628201704, 19.345876092281483,
      29.18785887899524]
kd_shift = [14.294088756514212, 22.986498865932045, 29.443323254142815, 12.712517295897396, 17.92108047709987,
            24.076646736924463]
kd_ours = [14.294088756514212, 22.163548255607147, 16.868135418579232, 12.717746105020382, 18.230207132020986,
           21.445383216938087]

kd_ours_wn = [14.294088756514212, 30.611456927802234, 19.097269225031816, 14.172368680382245, 18.99625766573164,
              29.941348943341907]

kd_shift_wn = [14.294088756514212, 27.87546139482088, 28.737537780111182, 14.877350879111473, 18.121000679910633,
               30.370601709828765]

kd_std = np.std(kd)
kd_shift_std = np.std(kd_shift)
kd_ours_std = np.std(kd_ours)
kd_shift_wn_std = np.std(kd_shift_wn)
kd_ours_wn_std = np.std(kd_ours_wn)

fig, axs = plt.subplots(2, 1, figsize=(6, 5))
axs[0].plot(kd, label='KD', marker='o', linestyle='-', color='darkblue')
axs[0].plot(kd_shift, label='TED - GVU', marker='o', linestyle='-', color='darkorange')
axs[0].plot(kd_ours, label='TED (Ours)', marker='o', linestyle='-', color='darkgreen')
axs[0].fill_between(x, np.array(kd_ours) - kd_ours_std, np.array(kd_ours) + kd_ours_std, alpha=0.2, color='darkgreen')
axs[0].set_xticks([])
axs[0].set_ylabel('AVG HD95', fontproperties=font_prop)
axs[0].legend(loc='lower left', prop=font_prop_small)

axs[1].plot(kd, label='KD', marker='o', linestyle='-', color='darkblue')
axs[1].plot(kd_shift_wn, label='KD + AKT', marker='o', linestyle='-', color='darkred')
axs[1].plot(kd_ours_wn, label='KD + AKT + GVU', marker='o', linestyle='-', color='darkviolet')
axs[1].fill_between(x, np.array(kd_ours_wn) - kd_ours_wn_std, np.array(kd_ours_wn) + kd_ours_wn_std, alpha=0.2,
                    color='darkviolet')
x = range(0, 6, 1)
plt.xticks(x, ('RUNMC', 'BMC', 'I2CVB', 'UCL', 'BIDMC', 'HK',), fontproperties=font_prop)
axs[1].set_ylabel('AVG HD95', fontproperties=font_prop)
axs[1].legend(loc='lower left', prop=font_prop_small)

handles, labels = axs[0].get_legend_handles_labels()

plt.tight_layout()
plt.savefig('prostate_hausdorff.png')
plt.show()

plt.figure(figsize=(24, 1))
# Create custom artists for legend
KD = plt.Line2D((0, 1), (0, 0), color='darkblue', marker='o', linestyle='-')
ED_GVU = plt.Line2D((0, 1), (0, 0), color='darkorange', marker='o', linestyle='-')
ED = plt.Line2D((0, 1), (0, 0), color='darkgreen', marker='o', linestyle='-')
KD_AKT = plt.Line2D((0, 1), (0, 0), color='darkred', marker='o', linestyle='-')
KD_AKT_GVU = plt.Line2D((0, 1), (0, 0), color='darkviolet', marker='o', linestyle='-')

# Create legend from custom artist/label lists
plt.legend([KD, ED_GVU, ED, KD_AKT, KD_AKT_GVU],
           ['KD', 'TED - GVU', 'TED (Ours)', 'KD + AKT', 'KD + AKT + GVU'],
           prop=font_prop,
           ncol=5,
           loc='upper center')

plt.axis('off')
plt.tight_layout()
plt.savefig('legend_GVU.png')
plt.show()
