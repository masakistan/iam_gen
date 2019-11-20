import sys
import cv2
from os import makedirs
from os.path import join
import random
import numpy as np
from tqdm import tqdm

words_dir = sys.argv[1]
words_gt = sys.argv[2]

lines_dir = sys.argv[3]
lines_gt = sys.argv[4]

out_dir = sys.argv[5]
n_samples = int(sys.argv[6])
n_blank_lines = int(sys.argv[7])

try:
    makedirs(out_dir)
except:
    print("outdirs may already exist")

def parse_iam_gt(fpath, img_dir, load_images=False):
    data = {}
    with open(fpath, 'r') as fh:
        for line in tqdm(fh):
            if line[0] == '#':
                continue
            line = line.strip()
            d = line.split()
            lid, gt = d[0], d[-1].split('|')
            img_path = lid.split('-')
            img_path = join(img_dir, img_path[0], '-'.join(img_path[:2]), '-'.join(img_path) + '.png')
            if load_images:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                assert img is not None, "img {} is none".format(img_path)
            else:
                img = None
            data[lid] = (img, gt)
    return data

words_data = parse_iam_gt(words_gt, words_dir, load_images=True)
print("Done parsing words, found {} entries".format(len(words_data)))

lines_data = parse_iam_gt(lines_gt, lines_dir)
print("Done parsing lines, found {} entries".format(len(lines_data)))


dims = []
gts = []
for i in tqdm(range(n_samples)):
    rand_line_key = random.choice(list(lines_data))
    n_words = len(lines_data[rand_line_key][1])

    word_imgs = []
    word_gts = []
    spaces = np.random.normal(50, 25, n_words + 1).astype(int)
    spaces[spaces < 0] = 0
    width, height = np.sum(spaces), 0
    for j in range(n_words):
        rand_word_key = random.choice(list(words_data))
        word_img, word_gt = words_data[rand_word_key]
        word_imgs.append(word_img)
        word_gts.append(word_gt[0])

        cheight, cwidth, _ = word_img.shape

        if cheight > height:
            height = cheight
        width += cwidth
    height_add = np.random.normal(30, 10, 1).astype(int)[0]
    if height_add < 0:
        height_add = 0
    height += height_add + 10

    new_img = np.ones((height, width, 3)) * 255

    cur_x = 0
    for word, space in zip(word_imgs, spaces):
        h, w, c = word.shape

        cur_h = np.random.random_integers(0, height - h)
        new_img[cur_h : cur_h + h, cur_x + space : cur_x + space + w, :] = word
        cur_x += space + w

    out_path = join(out_dir, str(i) + '.png')
    cv2.imwrite(out_path, new_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    dims.append(new_img.shape)

    gts.append((out_path, ' '.join(word_gts)))

offset = len(gts)
for i in range(n_blank_lines):
    i += offset
    dim = random.choice(dims)
    new_img = np.ones(dim) * 255
    out_path = join(out_dir, str(i) + '.png')
    cv2.imwrite(out_path, new_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    gts.append((out_path, ''))
    
    
with open(join(out_dir, 'generated_labels.txt'), 'w') as fh:
    for gt in gts:
        line = "\t".join(gt)
        print(line)
        fh.write(line)
        fh.write("\n")
    
    
   
