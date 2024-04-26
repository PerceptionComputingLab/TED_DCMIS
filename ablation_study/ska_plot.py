import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

font_path = "times-new-roman.ttf"
font_prop = fm.FontProperties(fname=font_path, size=19)

# load data from csv
data1 = pd.read_csv("noise_augmentation/old_knowledge.csv")
data2 = pd.read_csv("noise_augmentation/old_knowledge_ska.csv")
data3 = pd.read_csv("noise_augmentation/old_knowledge_diffusion.csv")
data4 = pd.read_csv("noise_augmentation/old_knowledge_injection.csv")
data5 = pd.read_csv("noise_augmentation/old_knowledge_aug.csv")

# create figure
plt.figure(figsize=(20, 4))

plt.subplot(1, 5, 1)
plt.scatter(
    data2["x"],
    data2["y"],
    color="red",
    alpha=0.3,
    edgecolors="white",
)
# plt.title('(a)  ska-aug distillation', fontproperties=font_prop)
plt.xlabel("PC1", fontproperties=font_prop)
plt.ylabel("PC2", fontproperties=font_prop)

plt.subplot(1, 5, 2)
plt.scatter(
    data1["x"],
    data1["y"],
    color="blue",
    alpha=0.3,
    edgecolors="white",
)
# plt.title('(b) Naive knowledge distillation', fontproperties=font_prop)
plt.xlabel("PC1", fontproperties=font_prop)
plt.ylabel("PC2", fontproperties=font_prop)

plt.subplot(1, 5, 3)
plt.scatter(
    data4["x"],
    data4["y"],
    color="orange",
    alpha=0.3,
    edgecolors="white",
)
# plt.title('(c) Noise-injected distillation', fontproperties=font_prop)
plt.xlabel("PC1", fontproperties=font_prop)
plt.ylabel("PC2", fontproperties=font_prop)

plt.subplot(1, 5, 4)
plt.scatter(
    data3["x"],
    data3["y"],
    color="green",
    alpha=0.3,
    edgecolors="white",
)
# plt.title('(d) Diffusion-aug distillation', fontproperties=font_prop)
plt.xlabel("PC1", fontproperties=font_prop)
plt.ylabel("PC2", fontproperties=font_prop)

plt.subplot(1, 5, 5)
plt.scatter(
    data5["x"],
    data5["y"],
    color="purple",
    alpha=0.3,
    edgecolors="white",
)
# plt.title('(e) Augmentation-driven distillation', fontproperties=font_prop)
plt.xlabel("PC1", fontproperties=font_prop)
plt.ylabel("PC2", fontproperties=font_prop)

plt.tight_layout()
plt.savefig("old_knowledge_comparison.png", dpi=300)
plt.show()
