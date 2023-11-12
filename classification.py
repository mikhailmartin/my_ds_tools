import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def my_binary_classification_report(
    classifier,
    X: pd.DataFrame,
    y_true: pd.Series,
    threshold: float = .5,
    *,
    classifier_name: str | None = None,
    figsize: tuple[float, float] | None = None,
    cm_kw: dict | None = None,
    rc_kw: dict | None = None,
) -> None:
    """
    Предоставляет отчёт по бинарной классификации.

    Визуализирует матрицу ошибок, ROC- и Precision- и Recall- кривые, печатает
    `sklearn.classification_report` и индекс Gini.

    Args:
        classifier: классификатор.
        X: признаки.
        y_true: истинные метки.
        threshold: порог бинаризации.
        classifier_name: имя классификатора.
        figsize: (ширина, высота) рисунка в дюймах.
        cm_kw: словарь с именованными аргументами для
          sklearn.metrics.ConfusionMatrixDisplay.from_predictions().
        rc_kw: словарь с именованными аргументами для
          sklearn.metrics.RocCurveDisplay.from_predictions().
    """
    if figsize is None:
        figsize = (12.8, 10.6)
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    y_proba = classifier.predict_proba(X)[:, 1]
    y_pred = np.where(y_proba >= threshold, 1, 0)

    default_cm_kw = dict(colorbar=False)
    cm_kw = cm_kw or {}
    cm_kw = {**default_cm_kw, **cm_kw}
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes[0, 0], **cm_kw)
    axes[0, 0].set(title='Матрица ошибок')
    axes[0, 0].grid()

    sns.histplot(data=y_proba, stat='density', kde=True, ax=axes[0, 1])
    axes[0, 1].set(xlabel='Probability')

    default_rc_kw = dict(color='orange')
    rc_kw = rc_kw or {}
    rc_kw = {**default_rc_kw, **rc_kw}
    RocCurveDisplay.from_predictions(
        y_true, y_proba, name=classifier_name, ax=axes[1, 0], **rc_kw)
    axes[1, 0].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes[1, 0].set(title='ROC-кривая', xlim=(-0.01, 1), ylim=(0, 1.01))

    # Графики зависимости Precision и Recall от порога бинаризации
    precision_recall_plot(y_true, y_proba, ax=axes[1, 1])
    axes[1, 1].set(title='Precision- и Recall-кривые')

    plt.show()

    print(classification_report(y_true, y_pred))

    # Gini_index: https://habr.com/ru/company/ods/blog/350440/
    print(f'Индекс Gini = {2 * roc_auc_score(y_true, y_proba) - 1}')


def my_multiclass_classification_report(
    classifier,
    X: pd.DataFrame,
    y_true: pd.Series,
    *,
    figsize: tuple[float, float] | None = None,
    cm_kw: dict | None = None,
) -> None:
    """
    Предоставляет отчёт по мультиклассовой классификации.

    Args:
        classifier: классификатор.
        X: признаки.
        y_true: истинные метки.
        figsize: (ширина, высота) рисунка в дюймах.
        cm_kw: словарь с именованными аргументами для
          sklearn.metrics.ConfusionMatrixDisplay.from_predictions().
    """
    if figsize is None:
        figsize = (6.4, 4.8)
    fig, ax = plt.subplots(figsize=figsize)

    y_pred = classifier.predict(X)

    default_cm_kw = dict(colorbar=False)
    cm_kw = cm_kw or {}
    cm_kw = {**default_cm_kw, **cm_kw}
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, **cm_kw)
    ax.set(title='Матрица ошибок')

    plt.show()

    print(classification_report(y_true, y_pred))


def precision_recall_plot(
    y_true: pd.Series,
    y_proba: np.ndarray,
    ax: matplotlib.axes.Axes | None = None,
    nbin: int = 255,
) -> None:
    """
    Рисует на заданном matplotlib.axes.Axes графики зависимости Precision и Recall от
    порога бинаризации.

    Args:
        y_true: истинные метки.
        y_proba: предсказанные вероятности отнесения к положительному классу.
        ax: matplotlib.axes.Axes, на котором следует отрисовать графики.
        nbin: количество бинов для равночастотного биннинга.
    """
    if ax is None:
        ax = plt.subplot()

    # равночастотный биннинг
    thresholds = np.interp(np.linspace(0, len(y_proba), nbin+1), np.arange(len(y_proba)), np.sort(y_proba))[1: -1]

    threshold_len = len(thresholds)
    precision_scores = np.empty(threshold_len, dtype=float)
    recall_scores = np.empty(threshold_len, dtype=float)

    for i, threshold in enumerate(thresholds):
        y_pred = np.where(y_proba >= threshold, 1, 0)

        precision_scores[i] = precision_score(y_true, y_pred)
        recall_scores[i] = recall_score(y_true, y_pred)

    ax.plot(thresholds, precision_scores, color='red', label='precision')
    ax.plot(thresholds, recall_scores, color='blue', label='recall')
    ax.set(xlabel='величина порога')
    ax.legend()
    ax.grid()
