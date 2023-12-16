import os
import requests
from multiprocessing import Pool
from PIL import Image
from dotenv import dotenv_values
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


config = dotenv_values(".env")
ACCESS_KEY = config["ACCESS_KEY"]
UNSPLASH_ROOT = "https://api.unsplash.com"


def parse_url(url: str):
    return Image.open(requests.get(url, stream=True).raw)

def parse_urls_to_images(urls: list):
    images = Pool(min(16, len(urls))).map(parse_url, urls)
    return images


def parse_id(id: str):
    # e.g. https://unsplash.com/photos/Xh_yj0ZYKyA/download?force=true&w=360
    return Image.open(
        requests.get(
            f'https://unsplash.com/photos/{id}/download?force=true&h=720', 
            stream=True
        ).raw
    )

def parse_ids_to_images(ids: list):
    images = Pool(min(16, len(ids))).map(parse_id, ids)
    return images


def parse_query_to_ids(query: str, k: int = 3):
    """
    retrieves [k * 30] ids from querying unsplash api
    """
    # r = requests.get(f'http://10.17.5.237/photos.json')
    ret = []
    for i in range(k):
        url = f'{UNSPLASH_ROOT}/search/photos?query={query}&client_id={ACCESS_KEY}&page={i+1}&per_page=30'
        json_dict = requests.get(url).json()
        ret += [e['id'] for e in json_dict['results']]
    return ret

def gen_figure(images: list, title: str = None, k = 5):
    # returns a plt figure
    fig, axes = plt.subplots(max(k//4, 1), min(k, 4), figsize=(12, 4))
    ax: Axes = ax
    for image, ax in zip(images, axes.flatten()):
        ax.imshow(image)
        ax.axis("off")
    if title:
        fig.supxlabel(title)
    fig.tight_layout()
    return fig, axes