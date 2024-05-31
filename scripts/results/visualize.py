import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os


def inspect(dir: str):
    folders = os.listdir(dir)
    folders = [f for f in folders if 'FM_cls' in f]

    dataframes = []
    val_means = []
    test_means = []
    names = []
    for folder in folders:
        metrics = np.load(f"{dir}/{folder}/metrics.npy", allow_pickle=True).item()
        print(f"Folder: {folder}")

        
        for test_fold in metrics.keys():
            metrics[test_fold]['average_val_acc'] = np.mean([metrics[test_fold][f"val_fold_{val_fold}"]['final_val']['accuracy'] for val_fold in range(7)])
            metrics[test_fold]['average_val_f1'] = np.mean([metrics[test_fold][f"val_fold_{val_fold}"]['final_val']['f1_scores'] for val_fold in range(7)], axis=0)
            metrics[test_fold]['average_val_confusion_matrix'] = np.mean([metrics[test_fold][f"val_fold_{val_fold}"]['final_val']['confusion_matrix'] for val_fold in range(7)], axis=0)
            metrics[test_fold]['average_test_acc'] = np.mean([metrics[test_fold][f"val_fold_{val_fold}"]['test']['accuracy'] for val_fold in range(7)])
            metrics[test_fold]['average_test_f1'] = np.mean([metrics[test_fold][f"val_fold_{val_fold}"]['test']['f1_scores'] for val_fold in range(7)], axis=0)
            metrics[test_fold]['average_test_confusion_matrix'] = np.mean([metrics[test_fold][f"val_fold_{val_fold}"]['test']['confusion_matrix'] for val_fold in range(7)], axis=0)

        avg_val_accs = {'Validation Accuracy': [metrics[f]['average_val_acc'] for f in metrics.keys()]}
        avg_val_f1_aberrant = {'Validation F1-score aberrant': [metrics[f]['average_val_f1'][0] for f in metrics.keys()]}
        avg_val_f1_normal = {'Validation F1-score normal': [metrics[f]['average_val_f1'][1] for f in metrics.keys()]}
        avg_test_accs = {'Test Accuracy': [metrics[f]['average_test_acc'] for f in metrics.keys()]}
        avg_test_f1_aberrant = {'Test F1-score aberrant': [metrics[f]['average_test_f1'][0] for f in metrics.keys()]}
        avg_test_f1_normal = {'Test F1-score normal': [metrics[f]['average_test_f1'][1] for f in metrics.keys()]}
        
        val_metrics_df = pd.DataFrame({**avg_val_accs, **avg_val_f1_aberrant, **avg_val_f1_normal})
        test_metrics_df = pd.DataFrame({**avg_test_accs, **avg_test_f1_aberrant, **avg_test_f1_normal})

        figure = plt.figure()
        boxplot_val = pd.plotting.boxplot(val_metrics_df, showmeans=True, showfliers=False)
        # plt.xticks(rotation=45)
        plt.title(f"{folder} Validation Metrics")
        plt.savefig(f"{dir}/{folder}/boxplot_val.pdf")
        figure = plt.figure()
        boxplot_test = pd.plotting.boxplot(test_metrics_df, showmeans=True, showfliers=False)
        plt.title(f"{folder} Test Metrics")
        plt.savefig(f"{dir}/{folder}/boxplot_test.pdf")

        with open(f"{dir}/{folder}/val_table.txt", "w") as f:
            f.write(val_metrics_df.to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.2f}".format,))
        
        with open(f"{dir}/{folder}/test_table.txt", "w") as f:
            f.write(test_metrics_df.to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.2f}".format,))
            
        val_means.append(val_metrics_df.mean())
        test_means.append(test_metrics_df.mean())
        names.append(folder)


    val_means_df = pd.DataFrame(val_means)
    val_means_df.index = names
    print(val_means_df)
    with open(f"{dir}/all_experiments_val.txt", "w") as f:
        f.write(val_means_df.to_latex(formatters={"name": str.upper},
                  float_format="{:.2f}".format,))
    
    test_means_df = pd.DataFrame(test_means)
    test_means_df.index = names

    with open(f"{dir}/all_experiments_test.txt", "w") as f:
        f.write(test_means_df.to_latex(formatters={"name": str.upper},
                  float_format="{:.2f}".format,))



    
if __name__ == "__main__":
    inspect("output/TimeFormer")
    inspect("output/SMNN")
    inspect("output/TimeConvNet")
    inspect("output/GCN_TimeFormer")


