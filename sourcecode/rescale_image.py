import glob
import os

from PIL import Image

katkam_path = 'katkam-scaled'

katkam_files = glob.glob(katkam_path + '/*.jpg')
if not os.path.exists('katkam-rescaled'):
    os.makedirs('katkam-rescaled')
for file in katkam_files:
    img = Image.open(file)
    new_width = 128
    new_height = 92
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    
    file_name = os.path.basename(file)
    img.save('katkam-rescaled/' + file_name)
