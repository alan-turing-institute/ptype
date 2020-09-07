import glob


def get_datasets():
    return [file.split("/")[-1] for file in glob.glob("data/*.csv")]
