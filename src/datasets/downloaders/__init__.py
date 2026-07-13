"""
Dataset download/export utilities.

This package contains downloader/export scripts for audio-text and image-text
datasets used by the project.

Most modules can be run directly from the command line, for example:

    python -m src.datasets.downloaders.download_musiccaps --number 100
    python -m src.datasets.downloaders.download_mtg_jamendo --split train --number 100
    python -m src.datasets.downloaders.download_coco
    python -m src.datasets.downloaders.download_coco --skip-images --prepare-cxc
    python -m src.datasets.downloaders.download_clap_benchmark
"""

__all__ = [
    "download_audioset",
    "download_clap_benchmark",
    "download_coco",
    "download_emoset",
    "download_flickr30k",
    "download_mtg_jamendo",
    "download_musiccaps",
    "download_song_describer",
]
