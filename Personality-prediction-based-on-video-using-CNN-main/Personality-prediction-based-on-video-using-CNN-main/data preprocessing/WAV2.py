import os
from moviepy.editor import *

Path = "E:\\F Y P\\Personality prediction using image processing\\datasets\\First Impressions V2 (CVPR'17)\\validate\\Extracted\\validation videos\\"
archive = os.listdir(Path)

for file_name in archive:
        file_name = (file_name.split(".mp4"))[0]
        print(file_name)
        
        videoclip = VideoFileClip(Path + str(file_name) + ".mp4")
        audioclip = videoclip.audio
        audioclip.write_audiofile("E:\\F Y P\\Personality prediction using image processing\\datasets\\First Impressions V2 (CVPR'17)\\validate\\Extracted\\audio\\" + str(file_name) + ".wav")

