import PIL
from PIL import Image
import os

input_dir = "C:\_Dev\Conv-Satellite-Image-Recognition\datajpg\desert\\"
output_dir = "C:\_Dev\Conv-Satellite-Image-Recognition\datajpg2/desert/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for imageFile in os.listdir(input_dir):
    image = Image.open(input_dir + imageFile)
    image.resize((64, 64), Image.NEAREST).save(output_dir + imageFile)
