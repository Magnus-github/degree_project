import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os


def inspect(dir: str):
    folders = os.listdir(dir)
    folders = [f for f in folders if 'FM_classification' in f]

    dataframes = []
    means = []
    names = []
    for folder in folders:
        metrics = np.load(f"{dir}/{folder}/metrics.npy", allow_pickle=True).item()
        print(f"Folder: {folder}")
        print(metrics['fold_0']['best']['val_f1'])

        acc = {'Accuracy': [metrics[f]['best']['val_accuracy'] for f in metrics.keys()]}
        f1_score_aberrant = {'F1-score aberrant': [metrics[f]['best']['val_f1'][0] for f in metrics.keys()]}
        f1_score_normal = {'F1-score normal': [metrics[f]['best']['val_f1'][1] for f in metrics.keys()]}
        
        df = pd.DataFrame({**acc, **f1_score_aberrant, **f1_score_normal})

        figure = plt.figure()
        boxplot = pd.plotting.boxplot(df, showmeans=True, showfliers=False)
        # plt.xticks(rotation=45)
        plt.title(folder)
        plt.savefig(f"{dir}/{folder}/boxplot.pdf")

        with open(f"{dir}/{folder}/table.txt", "w") as f:
            f.write(df.to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.2f}".format,))
            
        means.append(df.mean())
        names.append(folder)

        dataframes.append(df)

    means = pd.DataFrame(means)
    means.index = names
    print(means)
    with open(f"{dir}/all_experiments.txt", "w") as f:
        f.write(means.to_latex(formatters={"name": str.upper},
                  float_format="{:.2f}".format,))



    
if __name__ == "__main__":
    inspect("output/")