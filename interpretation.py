import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def permutation_importance_plot(
    pi,
    feature_names,
    figsize=None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Отрисовывает Permutation Importance.

    Args:
        pi: результат sklearn.inspection.permutation_importance().
        feature_names: названия признаков.
        figsize: (ширина, высота) рисунка в дюймах.

    Returns:
        Кортеж `(fig, ax)`.
          fig: matplotlib.Figure, содержащий все графики.
          ax: matplotlib.axes.Axes с отрисованным графиком.
    """
    sorted_importances_idx = np.argsort(-pi.importances_mean)
    importances = pd.DataFrame(
        pi.importances[sorted_importances_idx].T,
        columns=feature_names[sorted_importances_idx],
    )

    if not figsize:
        figsize = (12, .3 * importances.shape[1])
    fig, ax = plt.subplots(figsize=figsize)

    sns.violinplot(importances, orient='h', ax=ax)
    ax.set(
        title='Permutation Importances',
        xlabel='Decrease in accuracy score',
    )
    ax.axvline(x=0, color='k', linestyle='--')

    return fig, ax
