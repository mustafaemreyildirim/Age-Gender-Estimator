'''
usage:
    face_resize.py  --im_path=<str>  --dest_path=<str>
'''


import os
import cv2
from tqdm import tqdm
from docopt import docopt


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def process_image(img):

    size = img.shape
    h, w = size[0], size[1]
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w / scale), int(h / scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, (min_side - new_w) / 2 + 1, (
                    min_side - new_w) / 2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, (min_side - new_w) / 2, (
                    min_side - new_w) / 2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side - new_h) / 2, (min_side - new_h) / 2, (min_side - new_w) / 2, (
                    min_side - new_w) / 2
    else:
        top, bottom, left, right = (min_side - new_h) / 2 + 1, (min_side - new_h) / 2, (min_side - new_w) / 2 + 1, (
                    min_side - new_w) / 2
    pad_img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])

    return pad_img


def main(args):

    dest_root = args['--dest_path']
    mkdir(dest_root)
    cwd = os.getcwd()  # delete '.DS_Store' existed in the args['im_path']
    os.chdir(args['--im_path'])
    os.system("find . -name '*.DS_Store' -type f -delete")
    os.chdir(cwd)

    if not os.path.isdir(dest_root):
        os.mkdir(dest_root)

    for subfolder in tqdm(os.listdir(args['--im_path'])):
        if not os.path.isdir(os.path.join(dest_root, subfolder)):
            os.mkdir(os.path.join(dest_root, subfolder))
        for image_name in os.listdir(os.path.join(args['--im_path'], subfolder)):
            print("Processing\t{}".format(os.path.join(args['--im_path'], subfolder, image_name)))
            img = cv2.imread(os.path.join(args['--im_path'], subfolder, image_name))
            if type(img) == type(None):
                print("damaged image %s, del it" % (img))
                os.remove(img)
                continue
            size = img.shape
            h, w = size[0], size[1]
            if max(w, h) > 512:
                img_pad = process_image(img)
            else:
                img_pad = img
            cv2.imwrite(os.path.join(dest_root, subfolder, image_name.split('.')[0] + '.jpg'), img_pad)


if __name__ == "__main__":
    min_side = 512
    args = docopt(__doc__)
    main(args)