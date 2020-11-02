from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_data(dataset_name, header=None):
    dataset_path = "../data/" + dataset_name + ".csv"
    return pd.read_csv(
        dataset_path,
        sep=",",
        encoding="ISO-8859-1",
        dtype=str,
        header=header,
        keep_default_na=False,
        skipinitialspace=True,
    )


def subsample_df(df, column_to_sample_from, sample_num):
    return df[[column_to_sample_from]].sample(n=sample_num, random_state=1)


def plot_bar(classes, values, title, xlabel=None, ylabel=None, y_lim_max=1.0):
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), classes)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0.0, y_lim_max)
    plt.title(title)
    plt.show()


def scatter_plot(y, y_hat):
    plt.figure(figsize=(12, 4))

    sorted_indices = sorted(range(len(y)),key=y.__getitem__)
    plt.scatter(range(len(y)), y[sorted_indices], label='Actual Price', s=10)
    plt.scatter(range(len(y)), y_hat[sorted_indices], label='Predicted Price', s=10)

    plt.title('Actual vs Fitted Values for Price')
    plt.xlabel('Item (sorted wrt actual price)')
    plt.ylabel('Price (in dollars)')
    plt.legend()
    plt.show()

def plot_column_type_posterior(p_t):
    # p_t has subtypes of date separately and is not ordered alphabetically
    posterior = OrderedDict()
    for t, p in sorted(p_t.items()):
        # maps subtypes to types (date-iso-8601 to date)
        t_0 = t.split("-")[0]

        # sum the subtypes of dates
        if t_0 in posterior.keys():
            posterior[t_0] += p
        else:
            posterior[t_0] = p

    if len(np.unique(list(posterior.values()))) == 1:
        posterior = OrderedDict([(t, 1 / len(posterior)) for t in posterior])

    plot_bar(
        posterior.keys(),
        posterior.values(),
        title="p(t|x): posterior dist. of column type",
        xlabel="type",
        ylabel="posterior probability",
    )


def plot_arff_type_posterior(
    arff_posterior, types=["date", "nominal", "numeric", "string"]
):
    plot_bar(
        types,
        arff_posterior,
        title="posterior dist. of column ARFF type",
        xlabel="type",
        ylabel="posterior probability",
    )


def plot_row_type_posterior(col, t="missing"):
    if t == "missing":
        i = 1
    elif t == "anomaly":
        i = 2
    plot_bar(
        col.unique_vals,
        col.p_z[col.type][:, i],
        title="p(z_i=" + t + "|t,x_i): posterior dist. for row type",
        xlabel="unique value",
        ylabel="posterior " + t + " probability",
    )


def confusion_matrix(annotations, predictions, classes):
    """Calculates the confusion matrix of a classifier.
    This function compares a set of true labels and a set of predicted labels
    obtained by a classifier.
    Parameters
    ----------
    annotations : array-like of shape (n_samples,)
        True labels.
    predictions : array-like of shape (n_samples,)
        Predicted labels.
    classes : array-like of shape (n_classes,)
        A list of possible class labels in alphabetical order.
    Returns
    -------
    confusion_matrix : {array-like} of shape (n_classes, n_classes)
        Confusion matrix. confusion_matrix[i,j] denotes the number of
        confusions where i and j respectively indicate the orders of
        the prediction and annotation.
    """
    n_classes = len(classes)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for annotation, prediction in zip(annotations, predictions):
        if prediction != "any":
            confusion_matrix[classes.index(prediction), classes.index(annotation),] += 1
    return confusion_matrix


def plot_confusion_matrix(
    confusion_matrix, classes, figure_path="confusion_matrix.png"
):
    """Plots a given confusion matrix.
    This function plots a given confusion matrix and saves it as a figure.
    Parameters
    ----------
    confusion_matrix : array-like of shape (n_classes, n_classes)
        Confusion matrix.
    classes : array-like of shape (n_classes,)
        Classes.
    figure_path : str, default=confusion_matrix.png
        Figure path.
    Returns
    -------
        None
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5),)
    fig.tight_layout()

    im = heatmap(
        confusion_matrix,
        classes,
        classes,
        x_label="True Class",
        y_label="Predicted Class",
        ax=ax,
        vmin=0,
        vmax=0,
    )
    annotate_heatmap(im, valfmt="{x:d}", size=20, textcolors=["black", "white"])
    fig.tight_layout()

    plt.show()
    # plt.savefig(figure_path, dpi=300, bbox_inches="tight")


# Heatmap and annotate_annotate are adapted from https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(
    data,
    row_labels,
    col_labels,
    ax=None,
    colorbar=None,
    x_label=None,
    y_label=None,
    vmin=-1.0,
    vmax=1.0,
    rotation=0,
    **kwargs,
):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.gca()

    # Plot the heatmap
    im = ax.imshow(
        data, cmap=plt.cm.gray_r, vmax=vmax, vmin=vmin, aspect="auto", **kwargs
    )
    if colorbar is not None:
        fig.colorbar(im, ax=ax)

    # # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    # ... and label them with the respective list entries.
    if rotation == 0.0:
        ax.set_xticklabels(col_labels, fontsize=15)
    else:
        ax.set_xticklabels(
            col_labels,
            fontsize=15,
            rotation=rotation,
            ha="left",
            rotation_mode="anchor",
        )
    ax.set_yticklabels(row_labels, fontsize=15)

    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=20)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=20)
    ax.xaxis.set_label_position("top")

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(True, which="minor")
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=["black", "white"],
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.
    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
