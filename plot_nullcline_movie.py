import os
import moviepy.video.io.ImageSequenceClip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.rcParams['figure.dpi'] = 150


# reference --- https://zulko.github.io/moviepy/ref/VideoClip/VideoClip.html
#image_folder='folder_with_images'
fps = 1
#img_folder = f"C:/Users/antho/Documents/Projects/Infodemics/Code/figures"
img_folder = r'C:\Users\antho\Documents\Projects\Infodemics\Code\figures'


def make_movie_revised(par_type = 'risk', null_type = 'x2_x3'):
    """
    function to make a nullcline movie revised
    :return: nullcline move in .mp4 format
    """
    output_name = f"{par_type}_{null_type}"
    img_folder = r'C:\Users\antho\Documents\Projects\Infodemics\Code\figures\risk\x2_x3'
    image_files = [img_folder+'/'+img for img in os.listdir(img_folder) if img.endswith(".jpeg")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=20)
    clip.write_videofile(f"{output_name}.mp4")

make_movie_revised()



#make_movie(fps = 0.25, image_folder = img_folder, par_type = 'risk', null_type = 'x1_x2', output_name = 'my_movie')
#imgpath = r'C:\Users\antho\Documents\Projects\Infodemics\Code\figures'
#file = 'education_sg_sb_nullcline_'

null_bin = ['sg_sb', 'ib_v', 'sb_ib']
par_bin = ['education', 'risk']
for k, v0 in enumerate(par_bin):
    for j, w0 in enumerate(null_bin):
        print(k, j)
        #make_movie(bifurcation_type=w0, curve_type = v0 + '_again', fps = 1.25)
        #picture_creator(curve_type = v0, bifur_type=w0)


