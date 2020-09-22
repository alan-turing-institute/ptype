from collections import OrderedDict
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
        col.p_z[:, i],
        title="p(z_i=" + t + "|t,x_i): posterior dist. for row type",
        xlabel="unique value",
        ylabel="posterior " + t + " probability",
    )
