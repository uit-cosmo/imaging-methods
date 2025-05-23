from phantom import load_data_and_preprocess
from phantom.cond_av import *
import matplotlib.pyplot as plt

plt.style.use(["cosmoplots.default"])
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath} \usepackage{mathptmx} \usepackage{amssymb} "
)

shot = 1160616018
ds = load_data_and_preprocess(shot, 0.01, "../data", preprocessed=True)

refx, refy = 6, 5

win_size = 120
threshold = 2.5
events, average, _ = find_events_and_2dca(
    ds,
    refx,
    refy,
    threshold=threshold,
    window_size=win_size,
    check_max=1,
    single_counting=True,
)

# matplotlib.style.use("cosmoplots.default")
fig, axes = plt.subplots(
    3, 3, figsize=(3 * 3.37, 3 * 2.08277), sharex=True, sharey=True
)
e = events[1]
peak = e.isel(x=refx, y=refy).frames.max().item()
dt = 5e-7

for i in [-1, 0, 1]:
    for j in [-1, 0, 1]:
        ax = axes[j + 1, i + 1]
        pixel = e.isel(x=refx + i, y=refy - j)
        ax.plot(e.time.values, pixel.frames.values, color="blue")
        R, Z = pixel.R.item(), pixel.Z.item()
        ax.hlines(
            2.5,
            pixel.time.min().item(),
            pixel.time.max().item(),
            color="black",
            linewidth=0.5,
        )
        ax.set_xticks(
            [-win_size / 4 * dt, 0, win_size / 4 * dt],
            labels=[r"$-t_\text{win}/2$", r"$0$", r"$+t_\text{win}/2$"],
            fontsize=10,
        )
        ax.set_yticks([0, threshold], labels=[r"$0$", r"$T_\text{2DCA}$"], fontsize=10)
        ax.vlines(0, -1, 2 * peak, linewidth=0.5, color="black")
        # ax.set_title(r"$R = {:.2f}, Z = {:.2f}$".format(R, Z))
        ax.set_ylim(-1, peak * 1.1)
        ax.fill_between(
            [-win_size / 4 * dt, win_size / 4 * dt], -2, 2 * peak, color="lightgrey"
        )

plt.savefig("event_illustration.eps", bbox_inches="tight")
plt.show()
