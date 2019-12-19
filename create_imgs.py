from PIL import Image
import requests
from io import BytesIO
import os
import glob
import shutil
import urllib.request
import csv
LOGO_NAME = 'coca-cola' #redbull, coca-cola, heineken
OUTPUT_FILENAME = 'cleaned_imgs_v4'
INPUT_FILENAME = 'LogosInTheWild-v2/cleaned-data/voc_format/' + LOGO_NAME + '/urls.txt'
def create_img_folder(output_file):
    #os.makedirs(output_file)
    file = open(INPUT_FILENAME, 'r').read().split('\n')
    print(file)
    for i in range(len(file)):
        splt = file[i].split('\t')
        print(splt[1])
        URL = str(splt[1])
        print(i)
        try:
            with urllib.request.urlopen(URL) as url:
                with open(OUTPUT_FILENAME + '/' + '{:06d}'.format(i+654) + '.png', 'wb') as f:
                    f.write(url.read())
        except:
            print('Bad Image!')
        #img = Image.open('cleaned_imgs/temp.jpg')
        '''response = requests.get(splt[1], stream=True)
        print(response)
        response.raw.decode_content = True
        img = Image.open(response.raw)
            #img = Image.open(BytesIO(response.content))
        img.show()
            #img.save(OUTPUT_FILENAME + '/' + '{:6d}'.format(i) + '.jpg')'''

def conv_png_to_jpg(output_file):
    os.makedirs(output_file)
    i = 0
    for img in glob.glob('cleaned_imgs/*.png'):
        print("Parsing %s" % img)
        try:
            image = Image.open(img)
            #print(OUTPUT_FILENAME + '/' + str(img)[-10:-4] + '.jpg')
            rgb_im = image.convert('RGB')
            rgb_im.save(OUTPUT_FILENAME + '/' + str(img)[-10:-4] + '.jpg')
        except:
            print("Can't open image.")
            #i -= 1
        #i += 1

def norm_imgs(output_file):
    os.makedirs(output_file)
    bad_imgs = set()
    for img in glob.glob('cleaned_imgsv2/*.jpg'):
        print('Parsing %s' % img)
        image = Image.open(img)
        if image.size[0] != 80 or image.size[1] != 80:
            new_img = image.resize((1920, 1200))
            print(OUTPUT_FILENAME + '/' + str(img)[-10:])
            new_img.save(OUTPUT_FILENAME + '/' + str(img)[-10:])
        else:
            print(str(img)[-10:])
            bad_imgs.add(str(img)[-10:])

    labels_rows = []
    with open('LogosIW-train/labelsv3.csv', 'r') as f:
        csvrows = csv.reader(f)
        for row in csvrows:
            print(row)
            if row[5] not in bad_imgs and int(row[1]) < 1920 and int(row[2]) < 1200 and int(row[3]) < 1920 and int(row[4]) < 1200:
                labels_rows.append(row)
                #scalex, scaley = Image.open('cleaned_imgs_v4/' + str(row[5])).size
                #print(scalex, scaley)
                #print([row[0], round(1920/scalex)*int(row[1]), round(1200/scaley)*int(row[2]), round(1920/scalex)*int(row[3]), round(1200/scaley)*int(row[4]), row[5], row[6]])
                #print([row[0], int(1920 * int(row[1])/scalex), int(1200 * int(row[2])/scaley), int(1920 * int(row[3])/scalex), int(1200 * int(row[4])/scaley), row[5], row[6]])
                #labels_rows.append([row[0], int(1920 * int(row[1])/scalex), int(1200 * int(row[2])/scaley), int(1920 * int(row[3])/scalex), int(1200 * int(row[4])/scaley), row[5], row[6]])
    with open('LogosIW-train/labelsv4.csv', mode='w', newline='') as labels:
        labels = csv.writer(labels, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in labels_rows:
            #print(line)
            labels.writerow([line[0], line[1], line[2], line[3], line[4], line[5], line[6]])
#create_img_folder(OUTPUT_FILENAME)
#conv_png_to_jpg(OUTPUT_FILENAME)
norm_imgs(OUTPUT_FILENAME)