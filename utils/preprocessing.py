import os.path as osp
import glob
import pandas as pd
from PIL import Image
from tqdm import tqdm
import clip


N_PX = 224
RAW_TSV = 'meta/raw.tsv'
VALID_TSV = 'meta/valid.tsv'


def resize_image():
    files = glob.glob(f'../raw/*.jpg')
    for file in tqdm(files):
        im = Image.open(file)
        width, height = im.size
        x = width // 2
        y = height // 2
        a = min(x, y)
        im = im.crop((x-a, y-a, x+a, y+a)).resize((N_PX, N_PX))
        # Shows the image in image viewer
        im.save(file.replace('raw', 'data'))


def get_metadata() -> pd.DataFrame:
    """
    validate the image files and return a dataframe with valid image files
    """
    if osp.exists(VALID_TSV):
        return pd.read_csv(VALID_TSV, sep='\t', index_col=False)

    df = pd.read_csv(RAW_TSV, sep='\t')
    valid_ids = set(i[5:-4] for i in glob.glob('raw/*.jpg'))
    df = df[df['photo_id'].isin(valid_ids)]
    df.to_csv(VALID_TSV, sep='\t', index=False)
    return df


def get_path(photo_id: str) -> str:
    """
    convert the path to the image file to the photo_id
    """
    return f'data/{photo_id}.jpg'


def collect_image_sizes() -> pd.DataFrame:
    """
    return a dataframe with the image meta data
    """
    # for our image, fix `w`=640, `h` varies
    width, height = [], []
    for image in glob.glob('raw/*.jpg'):
        img = Image.open(image)
        width.append(img.width)
        height.append(img.height)
    return pd.DataFrame({'width': width, 'height': height})


def partition():
    """
    partition the dataset into train, validation, and test
    """
    df = get_metadata()
    df = df.sample(frac=1.0, random_state=32).reset_index(drop=True)
    n = len(df) // 10
    train = df.iloc[:-2*n]
    valid = df.iloc[-2*n:-n]
    test = df.iloc[-n:]
    train.to_csv('meta/train.tsv', sep='\t', index=False)
    valid.to_csv('meta/eval.tsv', sep='\t', index=False)
    test.to_csv('meta/test.tsv', sep='\t', index=False)


if __name__ == '__main__':
    # this shows how our image sizes vary
    # print(collect_image_sizes().value_counts())

    # df = get_metadata()
    # clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")
    # print("Input resolution:", clip_model.visual.input_resolution)
    partition()
