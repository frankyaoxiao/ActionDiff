from PIL import Image

import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img', type=str, default='test_image.png')
    args = parser.parse_args()

    img = Image.open(args.input_img)

    # convert to jpg and make sure it only has 3 channels (RGB)
    img = img.convert('RGB')

    # save the image
    img.save(args.input_img.replace('.png', '.jpg'))