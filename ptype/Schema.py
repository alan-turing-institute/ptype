import pandas as pd


class Schema:
    def __init__(
        self,
        df,
        cols,
    ):
        self.df = df
        self.cols = cols

    def show(self):
        df = self.df.iloc[0:0, :].copy()
        df.loc[0] = [col.type for _, col in self.cols.items()]
        df.loc[1] = [col.get_normal_values() for _, col in self.cols.items()]
        df.loc[2] = [col.get_missing_values() for _, col in self.cols.items()]
        df.loc[3] = [col.get_anomalous_values() for _, col in self.cols.items()]
        return df.rename(
            index={
                0: "type",
                1: "normal values",
                2: "missing values",
                3: "anomalous values",
            }
        )

    def show_ratios(self):
        df = self.df.iloc[0:0, :].copy()
        df.loc[0] = [col.type for _, col in self.cols.items()]
        df.loc[1] = [col.get_normal_ratio() for _, col in self.cols.items()]
        df.loc[2] = [col.get_missing_ratio() for _, col in self.cols.items()]
        df.loc[3] = [col.get_anomalous_ratio() for _, col in self.cols.items()]
        return df.rename(
            index={
                0: "type",
                1: "ratio of normal values",
                2: "ratio of missing values",
                3: "ratio of anomalous values",
            }
        )

    def as_normal(self):
        return lambda series: series.map(
            lambda v: v if v in self.cols[series.name].get_normal_values() else pd.NA
        )
