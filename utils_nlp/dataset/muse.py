# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
    Utility functions for downloading, extracting, and reading the
    Muse dataset dictionaries
    https://github.com/facebookresearch/MUSE
"""


import os
import pandas as pd

from utils_nlp.dataset.url_utils import extract_zip, maybe_download
from utils_nlp.dataset.preprocess import convert_to_unicode

TRAIN_URL = 'https://dl.fbaipublicfiles.com/arrival/dictionaries/{}-{}.0-5000.txt'
TEST_URL = 'https://dl.fbaipublicfiles.com/arrival/dictionaries/{}-{}.5000-6500.txt'


def load_pandas_df(local_cache_path=".", file_split="train", src_language="en", trg_language="de"):
    """Downloads and extracts the dataset files.

    Utilities information can be found `on this link <https://github.com/facebookresearch/MUSE/>`_.

    Args:
        local_cache_path (str, optional): Path to store the data.
            Defaults to "./".
        file_split (str, optional): The subset to load.
            One of: {"train", "test"}
            Defaults to "train".
        language (str, optional): language subset to read.
            One of the avaialble pairs from above URL
    Returns:
        pd.DataFrame: pandas DataFrame containing the specified
        word pairs.
    """

    if file_split == "train":
        url = TRAIN_URL.format(src_language, trg_language)
    elif file_split == "test":
        url = TEST_URL.format(src_language, trg_language)

    folder_name = local_cache_path + "/dictionaries"
    file_name = url.split("/")[-1]

    maybe_download(url, file_name, folder_name)

    with open(os.path.join(folder_name, file_name), "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    line_list = [line.split(" ") for line in lines]

    df = pd.DataFrame({"src_word": list(l[0] for l in line_list), "trg_word": list(l[1] for l in line_list)})

    return df
