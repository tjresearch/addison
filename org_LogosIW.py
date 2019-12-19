from PIL import Image
import requests
from io import BytesIO
import os
import glob
import shutil
import csv
import xml.etree.ElementTree as ET
import lxml.etree
# Designated Logos for this round: Heineken, RedBull, Coca-cola
LOGO_NAME = 'heineken'
INPUT_PATH = 'LogosInTheWild-v2/cleaned-data/voc_format/' + LOGO_NAME
OUTPUT_PATH = 'LogosIW-train/' + LOGO_NAME
OUTPUT_CSV = 'LogosIW-train/labelsv2.csv'
start_point = 0
def bounding_labels(output):
    file = open(INPUT_PATH + '/urls.txt').read().split('\n')
    print(file)
    lines = []
    start = 653
    valid_imgs = set()
    for img in glob.glob('cleaned_imgs_jpgs/*.jpg'):
        print("Parsing %s" % img)
        valid_imgs.add(str(img)[-10:-4])
    #print(valid_imgs)
    value = 475
    for i in range(len(file)):
        start += 1
        if "{:06d}".format(start) in valid_imgs:
            splt = file[i].split('\t')
            print(splt)
            root = lxml.etree.parse(INPUT_PATH + '/img' + splt[0] + '.xml')
            results = root.findall('object')
            xmins = [r.find('bndbox/xmin').text for r in results]
            ymins = [r.find('bndbox/ymin').text for r in results]
            xmaxs = [r.find('bndbox/xmax').text for r in results]
            ymaxs = [r.find('bndbox/ymax').text for r in results]
            print(xmins)
            for j in range(len(xmins)):
                lines.append([])
                lines[-1] = [start, xmins[j], ymins[j], xmaxs[j], ymaxs[j], "{:06d}".format(value)+'.jpg', LOGO_NAME]
            value += 1

    for line in lines:
        print(line)
    '''root = lxml.etree.parse(INPUT_PATH + '/img000005.xml')
    results = root.findall('object')
    textnumbers = [r.find('bndbox/xmin').text for r in results]
    print(textnumbers)'''
    with open(OUTPUT_CSV, mode='a', newline='') as labels:
        labels = csv.writer(labels, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in lines:
            print(line)
            labels.writerow([line[0], line[1], line[2], line[3], line[4], line[5], line[6]])
    print(start)
def img_processing(input_file, txt_file):
    os.makedirs(OUTPUT_PATH)

bounding_labels(OUTPUT_CSV)