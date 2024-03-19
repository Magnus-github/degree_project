import os
import argparse


def rename_vid_files(path, new_path):
    for file in os.listdir(path):
        id = file.split(" ")[1]
        if "_" in id:
            id = id.split("_")[0]
        
        count = 0
        for f in os.listdir(new_path):
            if id.zfill(4) in f:
                count += 1
        new_name = "ID_{:04d}_{}.pkl".format(int(id), count)

        os.rename(os.path.join(path, file), os.path.join(new_path,new_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename video files in a directory")
    parser.add_argument("--path", type=str, default="/Midgard/Data/tibbe/datasets/own/pose_sequences_openpose", help="Path to the directory with the video files")
    args = parser.parse_args()

    new_path = new_path = "/".join(args.path.split("/")[:-1]) + "/pose_sequences_openpose_renamed1/"
    print(new_path)
    print(args.path)
    os.makedirs(new_path, exist_ok=True)
    rename_vid_files(args.path, new_path)