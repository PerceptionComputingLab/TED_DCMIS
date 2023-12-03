import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

'''
    dataset: mm-i (LV-endo)
    approach: kd, kd_shift, kd_ours, kd_ours_wn, kd_shift_wn
    approach: kd, kd + akt + gvu, kd + akt + gvu + ska, kd + akt + ska, kd + akt
    data source: analysis/table_figure.py
'''

font_path = 'times-new-roman.ttf'
font_prop = fm.FontProperties(fname=font_path, size=19)
font_prop_small = fm.FontProperties(fname=font_path, size=10)
kd = [0.8791325426345765, 0.892169167759933, 0.8798588733696949, 0.853472241436826]
kd_shift = [0.8791325426345765, 0.8895649982526521, 0.8820620618005135, 0.873666650501003]
kd_ours = [0.8791325426345765, 0.8879488429782905, 0.8832923497628699, 0.8869112151842983]
kd_shift_wn = [0.8791325426345765, 0.8931023215671264, 0.8834791835130629, 0.8784230221839375]
kd_ours_wn = [0.8791325426345765, 0.8923314048300408, 0.8906503342716912, 0.8790639655531294]

kd_std = np.std(kd)
kd_shift_std = np.std(kd_shift)
kd_ours_std = np.std(kd_ours)
kd_shift_wn_std = np.std(kd_shift_wn)
kd_ours_wn_std = np.std(kd_ours_wn)

x = ['Siemens', 'Philips', 'GE', 'Canon']

fig, axs = plt.subplots(2, 1, figsize=(6, 5))
axs[0].plot(kd, label='KD', marker='o', linestyle='-', color='darkblue')
axs[0].plot(kd_shift, label='ED - GVU', marker='o', linestyle='-', color='darkorange')
axs[0].plot(kd_ours, label='ED (Ours)', marker='o', linestyle='-', color='darkgreen')
axs[0].fill_between(x, np.array(kd_ours) - kd_ours_std, np.array(kd_ours) + kd_ours_std, alpha=0.2, color='darkgreen')
axs[0].set_xticks([])
axs[0].set_ylabel('AVG DICE', fontproperties=font_prop)
axs[0].legend(loc='lower left', prop=font_prop_small)

axs[1].plot(kd, label='KD', marker='o', linestyle='-', color='darkblue')
axs[1].plot(kd_shift_wn, label='KD + AKT', marker='o', linestyle='-', color='darkred')
axs[1].plot(kd_ours_wn, label='KD + AKT + GVU', marker='o', linestyle='-', color='darkviolet')
axs[1].fill_between(x, np.array(kd_ours_wn) - kd_ours_wn_std, np.array(kd_ours_wn) + kd_ours_wn_std, alpha=0.2,
                    color='darkviolet')
x = range(0, 4, 1)
plt.ylabel("AVG DICE", fontproperties=font_prop)
plt.xticks(x, ('Siemens', 'Philips', 'GE', 'Canon',), fontproperties=font_prop)
axs[1].set_ylabel('AVG DICE', fontproperties=font_prop)
axs[1].legend(loc='lower left', prop=font_prop_small)

plt.tight_layout()
plt.savefig('mm_dice.png')
plt.show()

kd = [13.387890509885922, 10.420290327492628, 8.975877363622196, 21.266730908730928]
kd_shift = [13.387890509885922, 8.036472602385132, 11.608730817282728, 16.580847910266105]
kd_ours = [13.387890509885922, 8.034922057768501, 10.456572888817076, 13.916535674201294]
kd_shift_wn = [13.387890509885922, 7.985920559182697, 12.352522008542692, 19.68628393579789]
kd_ours_wn = [13.387890509885922, 9.306478961919558, 11.832571173633633, 18.30693681159049]

kd_std = np.std(kd)
kd_shift_std = np.std(kd_shift)
kd_ours_std = np.std(kd_ours)
kd_shift_wn_std = np.std(kd_shift_wn)
kd_ours_wn_std = np.std(kd_ours_wn)

fig, axs = plt.subplots(2, 1, figsize=(6, 5))
axs[0].plot(kd, label='KD', marker='o', linestyle='-', color='darkblue')
axs[0].plot(kd_shift, label='ED - GVU', marker='o', linestyle='-', color='darkorange')
axs[0].plot(kd_ours, label='ED (Ours)', marker='o', linestyle='-', color='darkgreen')
axs[0].fill_between(x, np.array(kd_ours) - kd_ours_std, np.array(kd_ours) + kd_ours_std, alpha=0.2, color='darkgreen')
axs[0].set_xticks([])
axs[0].set_ylabel('AVG HD95', fontproperties=font_prop)
axs[0].legend(loc='upper left', prop=font_prop_small)

axs[1].plot(kd, label='KD', marker='o', linestyle='-', color='darkblue')
axs[1].plot(kd_shift_wn, label='KD + AKT', marker='o', linestyle='-', color='darkred')
axs[1].plot(kd_ours_wn, label='KD + AKT + GVU', marker='o', linestyle='-', color='darkviolet')
axs[1].fill_between(x, np.array(kd_ours_wn) - kd_ours_wn_std, np.array(kd_ours_wn) + kd_ours_wn_std, alpha=0.2,
                    color='darkviolet')
x = range(0, 4, 1)
plt.xticks(x, ('Siemens', 'Philips', 'GE', 'Canon',), fontproperties=font_prop)
axs[1].set_ylabel('AVG HD95', fontproperties=font_prop)
axs[1].legend(loc='upper left', prop=font_prop_small)

handles, labels = axs[0].get_legend_handles_labels()

plt.tight_layout()
plt.savefig('mm_hausdorff.png')
plt.show()
