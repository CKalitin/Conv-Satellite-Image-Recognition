import sys
import os
from PIL import Image

source_folder = "C:\_Dev\Conv-Satellite-Image-Recognition\datapng\\water\\"
destination_folder = "C:\_Dev\Conv-Satellite-Image-Recognition\datajpg\\water\\"

if not os.path.exists(destination_folder):
    # destination_folder does not exist
    os.makedirs(destination_folder)

for imagefile in os.listdir(source_folder):
    img = Image.open(f'{source_folder}{imagefile}')
    #img = img.convert('RGB')
    clean_imgname = os.path.splitext(imagefile)[0]
    img.save(f'{destination_folder}{clean_imgname}.jpg', 'jpeg')
    print('Conversion done!')
