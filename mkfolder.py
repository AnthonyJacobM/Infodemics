import os

# reference:
# https://hinty.io/brucehardywald/how-to-create-folder-in-python/#:~:text=To%20create%20a%20folder%20in%20python%20script%20near,3%20Call%20os.mkdir%20passing%20formed%20an%20absolute%20path

# create a new folder for the figures!
os.mkdir('figures')

script_path = os.path.realpath(__file__)
new_abs_path = os.path.join(script_path, 'figures')
if not os.path_exists(new_abs_path):
    os.mkdir(new_abs_path)