import os
import glob
from tqdm import tqdm
from PIL import Image,ImageFile
from joblib import Parallel,delayed

def resize_image(img_path,output_folder,new_size):
    base_name = os.path.basename(img_path)
    outpath = os.path.join(output_folder,base_name)
    img = Image.open(img_path)
    img = img.resize(
        (new_size[0],new_size[1]),resample=Image.BILINEAR
    )
    img.save(outpath)

if __name__ == '__main__':
    input_folder = '../input/'
    output_folder = '/kaggle/working/train64'
    images = glob.glob(os.path.join(input_folder,"*.jpg"))
    Parallel(n_jobs=12){
        delayed(resize_image)(
            i,
            output_folder,
            (r,c)
        ) for i in tqdm(images)
    }
