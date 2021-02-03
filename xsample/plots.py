import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

string_cmap = "div yel grn"
cmap = mpl.cm.get_cmap(string_cmap)
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[cmap(0.2), "k", "red"]) 
#plt.rcParams['text.usetex'] = True

def SamplesCornerPlot(arrs, paramnames):
    if not isinstance(arrs, list):
        arrs = [arrs]
    nparams = arrs[0].shape[-1]
    fig = plt.figure(figsize=(nparams * 1.5, nparams * 1.5))
    width_ratios = [1] * nparams
    height_ratios = [1] * nparams
    spec = fig.add_gridspec(ncols=nparams, nrows=nparams, width_ratios=width_ratios, height_ratios=height_ratios, wspace=0, hspace=0)
    
    xlims = []
    for i in range(nparams):
        for j in range(i + 1, nparams):
            ax = fig.add_subplot(spec[j, i])
            for arr in arrs:
                sns.kdeplot(x=arr[:, i], y=arr[:, j], ax=ax, alpha=0.4)
            if j == nparams - 1:
                ax.set_xlabel(paramnames[i])
                xlims.append(ax.get_xlim())
            if i == 0:
                ax.set_ylabel(paramnames[j])
    axes = fig.get_axes()
    for i in range(nparams):
        ax = fig.add_subplot(spec[i, i])
        for arr in arrs:
            sns.kdeplot(data=arr[:, i], ax=ax, alpha=0.5)
        if not i == nparams -1:
            ax.set_xlim(xlims[i])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    for ax in fig.get_axes():
        ax.tick_params(axis='both', rotation=45, direction='inout')
        if not ax.is_first_col():
            ax.set_yticklabels([])
        if not ax.is_last_row():
            ax.set_xticklabels([])

    return fig, ax

def SamplesMarginalsPlot(arrs, paramnames):
    if not isinstance(arrs, list):
        arrs = [arrs]
    nparams = arrs[0].shape[-1]
    fig = plt.figure(figsize=(nparams))