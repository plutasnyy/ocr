import warnings
from pathlib import Path

import click
from PIL import Image

from model import MNISTModel

warnings.simplefilter("ignore")


@click.group()
def ocr():
    pass


@ocr.command()
@click.option('--path', required=True, type=click.Path())
def predict(path):
    model = MNISTModel.load_from_checkpoint('weights.ckpt')

    predictions = []
    path = Path(path)

    if path.is_file():
        paths = [path]
    else:
        paths = filter(lambda x: x.is_file(), path.rglob('*'))

    for path in paths:
        try:
            image = Image.open(path)
            pred = model.predict_image(image)
            predictions.append([str(path), pred])
        except Exception as e:  # not best practise to catch everything
            print('ERROR', path, 'Doesn"t work')

    for filename, pred in predictions:
        print(f'{filename};{pred}')


if __name__ == '__main__':
    ocr()
