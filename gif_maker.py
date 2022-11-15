import glob
from PIL import Image


fp_in = "./*Both*.png"
fp_out = "mm_Ha_corr_6_Panels_new_box.gif"
import os
import imageio
'''
png_dir = fp_in
images = []
for file_name in sorted(os.listdir(png_dir)):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave(fp_out, images)
'''
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=50, loop=0)

