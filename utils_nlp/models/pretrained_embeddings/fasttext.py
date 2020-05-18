# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Functions to help users load and extract fastText pretrained embeddings."""

import os
import zipfile

from gensim.models.keyedvectors import KeyedVectors

from utils_nlp.dataset.url_utils import maybe_download
from utils_nlp.models.pretrained_embeddings import FASTTEXT_EN_URL

def _download_fasttext_vectors(download_dir, file_name="wiki.simple.vec"):
    """ Downloads pre-trained word vectors for English, trained on Wikipedia using
    fastText. You can directly download the vectors from here:
    https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec

    For the full version of pre-trained word vectors, change the url for
    FASTTEXT_EN_URL to https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
    in __init__.py

    Args:
        download_dir (str): File path to download the file
        file_name (str) : File name given by default but can be changed by the user.

    Returns:
        str: file_path to the downloaded vectors.
    """

    return maybe_download(
        FASTTEXT_EN_URL, filename=file_name, work_directory=download_dir
    )


def _maybe_download_and_extract(dest_path, file_name):
    """ Downloads and extracts fastText vectors if they don’t already exist

    Args:
        dest_path(str): Final path where the vectors will be extracted.
        file_name(str): File name of the fastText vector file.

    Returns:
        str: File path to the fastText vector file.
    """

    dir_path = os.path.join(dest_path, "fastText")
    file_path = os.path.join(dir_path, file_name)

    if not os.path.exists(file_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        print("Downloading to: ", dir_path, "File: ", file_name)
        file_path = _download_fasttext_vectors(dir_path, file_name=file_name)
    else:
        print("Vector file already exists. No changes made.", file_path)

    return file_path


def load_pretrained_vectors(dest_path, file_name="wiki.simple.vec", limit=None):
    """ Method that loads fastText vectors. Downloads if it doesn't exist.

    Args:
        file_name(str): Name of the fastText file.
        dest_path(str): Path to the directory where fastText vectors exist or will be
        downloaded.

    Returns:
        gensim.models.keyedvectors.Word2VecKeyedVectors: Loaded word2vectors

    """

    file_path = _maybe_download_and_extract(dest_path, file_name)
    model = KeyedVectors.load_word2vec_format(file_path, limit=limit)
    return model
