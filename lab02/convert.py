# coding=utf-8

import os

dir = 'selfie/input/'

for i in range(30):
    os.system('convert -resize 60x100! -colorspace Gray ' + dir + str(i) +
'_in.jpg ' + dir + str(i) + '.jpg')