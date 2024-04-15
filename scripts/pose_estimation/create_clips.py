import os
import numpy as np

LEN_CLIP = 30 # in seconds

def create_clips(file, fps, outfolder):
    sequence = np.load(file)
    T = sequence.shape[0]

    # cut the sequence to be divisible by LEN_CLIP*fps
    cut = T%(LEN_CLIP*fps)

    sequence = sequence[cut//2:-cut//2]
    T = sequence.shape[0]

    K = T//(LEN_CLIP*fps)

    for k in range(K):
        clip = sequence[k*LEN_CLIP*fps:(k+1)*LEN_CLIP*fps]
        np.save(os.path.join(outfolder, f"{file.split('/')[-1].split('.')[0]}_{k}.npy"), clip)



def main():
    data_folder = "/Midgard/Data/tibbe/datasets/own/poses_smooth_np"
    fps = 25
    outfolder = "/Midgard/Data/tibbe/datasets/own/clips_smooth_np"

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    for file in os.listdir(data_folder):
        create_clips(os.path.join(data_folder, file), fps, outfolder)

    # print(len(os.listdir(outfolder)))
    # for file in os.listdir(outfolder):
    #     sequence = np.load(os.path.join(outfolder, file))
    #     if sequence.shape[0] != LEN_CLIP*25:
    #         print(file)
    #         print("NUM_FRAMES: ", sequence.shape[0])


    
if __name__ == "__main__":
    main()
