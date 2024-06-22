import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import torch

from scipy import stats

import os

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
TICKS_FONTSIZE = 35
LABEL_FONTSIZE = 45

def get_val_and_test_metrics(metrics: dict, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_confs = [metrics[f]['average_val_confusion_matrix'] for f in metrics.keys()]
    occs_class0_val = [np.sum(conf[0]) for conf in val_confs]
    occs_class1_val = [np.sum(conf[1]) for conf in val_confs]
    mean_test_confs = [metrics[f]['average_test_confusion_matrix'] for f in metrics.keys()]
    occs_class0_test = [np.sum(conf[0]) for conf in mean_test_confs]
    occs_class1_test = [np.sum(conf[1]) for conf in mean_test_confs]
    # # mean_test_conf = np.mean([metrics[f]['average_test_confusion_matrix'] for f in metrics.keys()], axis=0)
    # occ_class0_test = np.sum(mean_test_conf[0])
    # occ_class1_test = np.sum(mean_test_conf[1])
    if mode  == '2class_wAbnormal':
        avg_val_accs = {'Accuracy': [metrics[f]['average_val_acc'] for f in metrics.keys()]}
        avg_val_f1_aberrant = {'F1-score aberrant': [metrics[f]['average_val_f1'][0] for f in metrics.keys()]}
        avg_val_f1_normal = {'F1-score normal': [metrics[f]['average_val_f1'][1] for f in metrics.keys()]}
        avg_test_accs = {'Accuracy': [metrics[f]['average_test_acc'] for f in metrics.keys()]}
        avg_test_f1_aberrant = {'F1-score aberrant': [metrics[f]['average_test_f1'][0] for f in metrics.keys()]}
        avg_test_f1_normal = {'F1-score normal': [metrics[f]['average_test_f1'][1] for f in metrics.keys()]}

        val_metrics_df = pd.DataFrame({**avg_val_accs, **avg_val_f1_aberrant, **avg_val_f1_normal})
        test_metrics_df = pd.DataFrame({**avg_test_accs, **avg_test_f1_aberrant, **avg_test_f1_normal})

        weighted_val_f1s = [(occ_class0_val * f1_abberrant + occ_class1_val * f1_normal)/(occ_class0_val + occ_class1_val) for occ_class0_val, occ_class1_val, f1_abberrant, f1_normal in zip(occs_class0_val, occs_class1_val, val_metrics_df['F1-score aberrant'], val_metrics_df['F1-score normal'])]
        val_metrics_df.loc[:, "Weigthed F1-score"] = weighted_val_f1s

        weighted_test_f1s = [(occ_class0_test * f1_abberrant + occ_class1_test * f1_normal)/(occ_class0_test + occ_class1_test) for occ_class0_test, occ_class1_test, f1_abberrant, f1_normal in zip(occs_class0_test, occs_class1_test, test_metrics_df['F1-score aberrant'], test_metrics_df['F1-score normal'])]
        test_metrics_df.loc[:, "Weigthed F1-score"] = weighted_test_f1s
    elif mode == '2class_noAbnormal':
        avg_val_accs = {'Accuracy': [metrics[f]['average_val_acc'] for f in metrics.keys()]}
        avg_val_f1_absent = {'F1-score absent': [metrics[f]['average_val_f1'][0] for f in metrics.keys()]}
        avg_val_f1_normal = {'F1-score normal': [metrics[f]['average_val_f1'][1] for f in metrics.keys()]}
        avg_test_accs = {'Accuracy': [metrics[f]['average_test_acc'] for f in metrics.keys()]}
        avg_test_f1_absent = {'F1-score absent': [metrics[f]['average_test_f1'][0] for f in metrics.keys()]}
        avg_test_f1_normal = {'F1-score normal': [metrics[f]['average_test_f1'][1] for f in metrics.keys()]}

        val_metrics_df = pd.DataFrame({**avg_val_accs, **avg_val_f1_absent, **avg_val_f1_normal})
        test_metrics_df = pd.DataFrame({**avg_test_accs, **avg_test_f1_absent, **avg_test_f1_normal})

        weighted_val_f1s = [(occ_class0_val * f1_absent + occ_class1_val * f1_normal)/(occ_class0_val + occ_class1_val) for occ_class0_val, occ_class1_val, f1_absent, f1_normal in zip(occs_class0_val, occs_class1_val, val_metrics_df['F1-score absent'], val_metrics_df['F1-score normal'])]
        val_metrics_df.loc[:, "Weigthed F1-score"] = weighted_val_f1s

        weighted_test_f1s = [(occ_class0_test * f1_absent + occ_class1_test * f1_normal)/(occ_class0_test + occ_class1_test) for occ_class0_test, occ_class1_test, f1_absent, f1_normal in zip(occs_class0_test, occs_class1_test, test_metrics_df['F1-score absent'], test_metrics_df['F1-score normal'])]
        test_metrics_df.loc[:, "Weigthed F1-score"] = weighted_test_f1s
    elif mode == '3class':
        avg_val_accs = {'Accuracy': [metrics[f]['average_val_acc'] for f in metrics.keys()]}
        avg_val_f1_absent = {'F1-score absent': [metrics[f]['average_val_f1'][0] for f in metrics.keys()]}
        avg_val_f1_abnormal = {'F1-score abnormal': [metrics[f]['average_val_f1'][1] for f in metrics.keys()]}
        avg_val_f1_normal = {'F1-score normal': [metrics[f]['average_val_f1'][2] for f in metrics.keys()]}
        avg_test_accs = {'Accuracy': [metrics[f]['average_test_acc'] for f in metrics.keys()]}
        avg_test_f1_absent = {'F1-score absent': [metrics[f]['average_test_f1'][0] for f in metrics.keys()]}
        avg_test_f1_abnormal = {'F1-score abnormal': [metrics[f]['average_test_f1'][1] for f in metrics.keys()]}
        avg_test_f1_normal = {'F1-score normal': [metrics[f]['average_test_f1'][2] for f in metrics.keys()]}

        val_metrics_df = pd.DataFrame({**avg_val_accs, **avg_val_f1_absent, **avg_val_f1_abnormal, **avg_val_f1_normal})
        test_metrics_df = pd.DataFrame({**avg_test_accs, **avg_test_f1_absent, **avg_test_f1_abnormal, **avg_test_f1_normal})

        occs_class2_val = [np.sum(conf[2]) for conf in val_confs]
        weighted_val_f1s = [(occ_class0_val * f1_absent + occ_class1_val * f1_abnormal + occ_class2_val * f1_normal)/(occ_class0_val + occ_class1_val + occ_class2_val) for occ_class0_val, occ_class1_val, occ_class2_val, f1_absent, f1_abnormal, f1_normal in zip(occs_class0_val, occs_class1_val, occs_class2_val, val_metrics_df['F1-score absent'], val_metrics_df['F1-score abnormal'], val_metrics_df['F1-score normal'])]
        val_metrics_df.loc[:, "Weigthed F1-score"] = weighted_val_f1s

        occs_class2_test = [np.sum(conf[2]) for conf in mean_test_confs]
        weighted_test_f1s = [(occ_class0_test * f1_absent + occ_class1_test * f1_abnormal + occ_class2_test * f1_normal)/(occ_class0_test + occ_class1_test + occ_class2_test) for occ_class0_test, occ_class1_test, occ_class2_test, f1_absent, f1_abnormal, f1_normal in zip(occs_class0_test, occs_class1_test, occs_class2_test, test_metrics_df['F1-score absent'], test_metrics_df['F1-score abnormal'], test_metrics_df['F1-score normal'])]
        test_metrics_df.loc[:, "Weigthed F1-score"] = weighted_test_f1s

    else:
        raise ValueError("Mode not recognized. Choose from '2class_wAbnormal','2class_noAbnormal', '3class'")

    return val_metrics_df, test_metrics_df


def get_ROC_from_confusion(confusion_matrix: np.ndarray):
    sensitivities = []
    specificities = []
    matrix_size = confusion_matrix.shape[0]
    idx = list(range(matrix_size))
    for i in range(matrix_size):
        no_i = [ind for ind in idx if ind!=i]
        TP = confusion_matrix[i][i]
        FP = np.sum(confusion_matrix[no_i][:,i])
        FN = np.sum(confusion_matrix[i][no_i])
        TN = np.sum(confusion_matrix[no_i][:,no_i])
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    if matrix_size == 2:
        sensitivities = {0: sensitivities[0], 1: sensitivities[1]}
        specificities = {0: specificities[0], 1: specificities[1]}
    elif matrix_size == 3:
        sensitivities = {0: sensitivities[0], 1: sensitivities[1], 2: sensitivities[2]}
        specificities = {0: specificities[0], 1: specificities[1], 2: specificities[2]}
    
    return sensitivities, specificities


def get_ROC_curves(metrics: dict, mode: str):
    thresholds = np.linspace(0,1,11)
    if '3class' in mode:
        map = {1: 0, 4: 1, 12: 2}
    elif '2class' in mode:
        map = {1: 0, 4: 0, 12: 1}
    
    mean_sensitivities = {}
    mean_specificities = {}
    
    for thresh in thresholds:
        sensitivities = []
        specificities = []
        for test_fold in metrics.keys():
            for val_fold in metrics[test_fold].keys():
                outprobs_labels = metrics[test_fold][val_fold]['final_val']['outputs']
                outprobs = outprobs_labels[:, :2]
                labels = outprobs_labels[:, 2]
                thresholded = (outprobs > thresh).int()
                predictions = thresholded.argmax(axis=1)
                labels = torch.tensor([map[label.item()] for label in labels])

                TP = 0
                FP = 0
                FN = 0
                TN = 0
                for i in range(len(predictions)):
                    if predictions[i] == 0 and labels[i] == 0:
                        TP += 1
                    elif predictions[i] == 0 and labels[i] == 1:
                        FP += 1
                    elif predictions[i] == 1 and labels[i] == 0:
                        FN += 1
                    else:
                        TN += 1
                
                sensitivity = TP / (TP + FN)
                specificity = TN / (TN + FP)
                sensitivities.append(sensitivity)
                specificities.append(specificity)
            
        mean_sensitivity = np.mean(sensitivities)
        mean_specificity = np.mean(specificities)
        mean_sensitivities[thresh] = mean_sensitivity
        mean_specificities[thresh] = mean_specificity

    return mean_sensitivities, mean_specificities

    









def plot_confusion_matrix(confusion_matrix: np.ndarray, name: str, dir: str, folder: str, mode: str, val_res:  str = 'final_val'):
    if mode == '2class_wAbnormal':
        figure = plt.figure(figsize=(15, 10))
        plt.imshow(confusion_matrix, cmap='viridis', vmin=0, vmax=1)
        cb=plt.colorbar()
        cb.ax.tick_params(labelsize=TICKS_FONTSIZE)
        # plt.title(f"{name} Confusion Matrix", fontsize=30)
        plt.xticks([0, 1], ['Aberrant', 'Normal'], fontsize=TICKS_FONTSIZE)
        plt.yticks([0, 1], ['Aberrant', 'Normal'], fontsize=TICKS_FONTSIZE)
        plt.ylabel('True label', fontsize=LABEL_FONTSIZE)
        plt.xlabel('Predicted label', fontsize=LABEL_FONTSIZE)
        plt.tight_layout()
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i, round(confusion_matrix[i, j],2), ha='center', va='center', fontsize=TICKS_FONTSIZE)
        plt.savefig(f"{dir}/{folder}/{val_res}_confusion_matrix.pdf")
    elif mode == '2class_noAbnormal':
        figure = plt.figure(figsize=(15, 10))
        plt.imshow(confusion_matrix, cmap='viridis', vmin=0, vmax=1)
        cb=plt.colorbar()
        cb.ax.tick_params(labelsize=TICKS_FONTSIZE)
        # plt.title(f"{name} Confusion Matrix", fontsize=30)
        plt.xticks([0, 1], ['Absent', 'Normal'], fontsize=TICKS_FONTSIZE)
        plt.yticks([0, 1], ['Absent', 'Normal'], fontsize=TICKS_FONTSIZE)
        plt.ylabel('True label', fontsize=LABEL_FONTSIZE)
        plt.xlabel('Predicted label', fontsize=LABEL_FONTSIZE)
        plt.tight_layout()
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i, round(confusion_matrix[i, j],2), ha='center', va='center', fontsize=TICKS_FONTSIZE)
        plt.savefig(f"{dir}/{folder}/{val_res}_confusion_matrix.pdf")
    elif mode == '3class':
        figure = plt.figure(figsize=(15, 10))
        plt.imshow(confusion_matrix, cmap='viridis', vmin=0, vmax=1)
        cb=plt.colorbar()
        cb.ax.tick_params(labelsize=TICKS_FONTSIZE)
        # plt.title(f"{name} Confusion Matrix", fontsize=30)
        plt.xticks([0, 1, 2], ['Absent', 'Abnormal', 'Normal'], fontsize=TICKS_FONTSIZE)
        plt.yticks([0, 1, 2], ['Absent', 'Abnormal', 'Normal'], fontsize=TICKS_FONTSIZE)
        plt.ylabel('True label', fontsize=LABEL_FONTSIZE)
        plt.xlabel('Predicted label', fontsize=LABEL_FONTSIZE)
        plt.tight_layout()
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i, round(confusion_matrix[i, j],2), ha='center', va='center', fontsize=TICKS_FONTSIZE)
        plt.savefig(f"{dir}/{folder}/{val_res}_confusion_matrix.pdf")


def statistical_significance(folder1: str, folder2: str, mode: str = '2class_wAbnormal'):
    metrics1 = np.load(f"{folder1}/metrics.npy", allow_pickle=True).item()
    metrics2 = np.load(f"{folder2}/metrics.npy", allow_pickle=True).item()

    name1 = folder1.split("/")[-1].split("_")[2:]
    name2 = folder2.split("/")[-1].split("_")[2:]
    name = "_".join(name1) + "_vs_" + "_".join(name2)

    output_dir = f"output/statistical_significance/{mode}/{name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    val_res = 'final_val'
    for metrics in [metrics1, metrics2]:
        for test_fold in metrics.keys():
            metrics[test_fold]['average_val_acc'] = np.mean([metrics[test_fold][f"val_fold_{val_fold}"][val_res]['accuracy'] for val_fold in range(7)])
            metrics[test_fold]['average_val_f1'] = np.mean([metrics[test_fold][f"val_fold_{val_fold}"][val_res]['f1_scores'] for val_fold in range(7)], axis=0)
            metrics[test_fold]['average_val_confusion_matrix'] = np.mean([metrics[test_fold][f"val_fold_{val_fold}"][val_res]['confusion_matrix'] for val_fold in range(7)], axis=0)
            metrics[test_fold]['average_test_acc'] = np.mean([metrics[test_fold][f"val_fold_{val_fold}"]['test']['accuracy'] for val_fold in range(7)])
            metrics[test_fold]['average_test_f1'] = np.mean([metrics[test_fold][f"val_fold_{val_fold}"]['test']['f1_scores'] for val_fold in range(7)], axis=0)
            metrics[test_fold]['average_test_confusion_matrix'] = np.mean([metrics[test_fold][f"val_fold_{val_fold}"]['test']['confusion_matrix'] for val_fold in range(7)], axis=0)
    val_metrics1, test_metrics1 = get_val_and_test_metrics(metrics1, mode)
    val_metrics2, test_metrics2 = get_val_and_test_metrics(metrics2, mode)

    p_values = {}
    for col in val_metrics1.columns:
        differences = val_metrics1[col].values - val_metrics2[col].values
        shapiro_test = stats.shapiro(differences)
        if shapiro_test[1] < 0.05:
            print(f"Shapiro test for {col} not passed, p-value: {shapiro_test[1]}. Differences are not normally distributed.")
            u, p = stats.mannwhitneyu(val_metrics1[col], val_metrics2[col])
            print(f"Mann-Whitney U test for {col}: {p}")
            p_values[col] = p
        else:
            t, p = stats.ttest_rel(val_metrics1[col], val_metrics2[col])
            print(f"p-value for {col}: {p}")
            p_values[col] = p
    
    with open(f"{output_dir}/p_values.txt", "w") as f:
        f.write(str(p_values))


def inspect(dir: str):
    folders = os.listdir(dir)
    folders = [f for f in folders if 'FM_cls' in f]

    mean_conf_mats_val = []
    val_means = []
    val_stds = []
    test_means = []
    names = []
    for folder in folders:
        metrics = np.load(f"{dir}/{folder}/metrics.npy", allow_pickle=True).item()
        print(f"Folder: {folder}")

        val_res = 'final_val'
        for test_fold in metrics.keys():
            metrics[test_fold]['average_val_acc'] = np.mean([metrics[test_fold][f"val_fold_{val_fold}"][val_res]['accuracy'] for val_fold in range(7)])
            metrics[test_fold]['average_val_f1'] = np.mean([metrics[test_fold][f"val_fold_{val_fold}"][val_res]['f1_scores'] for val_fold in range(7)], axis=0)
            metrics[test_fold]['average_val_confusion_matrix'] = np.mean([metrics[test_fold][f"val_fold_{val_fold}"][val_res]['confusion_matrix'] for val_fold in range(7)], axis=0)
            metrics[test_fold]['average_test_acc'] = np.mean([metrics[test_fold][f"val_fold_{val_fold}"]['test']['accuracy'] for val_fold in range(7)])
            metrics[test_fold]['average_test_f1'] = np.mean([metrics[test_fold][f"val_fold_{val_fold}"]['test']['f1_scores'] for val_fold in range(7)], axis=0)
            metrics[test_fold]['average_test_confusion_matrix'] = np.mean([metrics[test_fold][f"val_fold_{val_fold}"]['test']['confusion_matrix'] for val_fold in range(7)], axis=0)

        if '3class' in folder:
            mode = '3class'
        else:
            if 'noAbnormal' in folder:
                mode = '2class_noAbnormal'
            else:
                mode = '2class_wAbnormal'

        val_metrics_df, test_metrics_df = get_val_and_test_metrics(metrics, mode)

        figure = plt.figure(figsize=(15, 10))
        name = "_".join(reversed(folder.split("_")[2:]))
        boxplot_val = pd.plotting.boxplot(val_metrics_df, showmeans=True, showfliers=False)
        tick_labels = val_metrics_df.columns.to_list()
        tick_labels = [label.replace(' ', '\n') for label in tick_labels]
        plt.xticks(range(1,len(tick_labels)+1), tick_labels, fontsize=TICKS_FONTSIZE)
        plt.yticks(fontsize=TICKS_FONTSIZE)
        plt.ylim(0, 1)
        # plt.title(f"{name} Validation Metrics", fontsize=30)
        plt.tight_layout()
        plt.savefig(f"{dir}/{folder}/boxplot_{val_res}.pdf")
# df.loc[:, df.columns != 'b']
        figure = plt.figure(figsize=(10, 5))
        boxplot_test = pd.plotting.boxplot(test_metrics_df, showmeans=True, showfliers=False)
        plt.xticks(fontsize=TICKS_FONTSIZE)
        plt.yticks(fontsize=TICKS_FONTSIZE)
        plt.ylim(0, 1)
        # plt.title(f"{name} Test Metrics", fontsize=20)
        plt.savefig(f"{dir}/{folder}/boxplot_test.pdf")

        with open(f"{dir}/{folder}/{val_res}_table.txt", "w") as f:
            f.write(val_metrics_df.to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.2f}".format,))
        
        with open(f"{dir}/{folder}/test_table.txt", "w") as f:
            f.write(test_metrics_df.to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.2f}".format,))
            
        test_means.append(test_metrics_df.mean())
        val_means.append(val_metrics_df.mean())
        val_stds.append(val_metrics_df.std())
        mean_val_conf = np.mean([metrics[f]['average_val_confusion_matrix'] for f in metrics.keys()], axis=0)
        mean_conf_mats_val.append(mean_val_conf)
        names.append(folder)

        norm_mean_val_conf = mean_val_conf / np.sum(mean_val_conf, axis=1)[:, np.newaxis]
        # plot confusion matrix
        plot_confusion_matrix(norm_mean_val_conf, name, dir, folder, mode, val_res)

        # plot ROC curve
        # sens_0 = [0]
        # spec_0 = [1]
        # val_conf_mats = [metrics[f]['average_val_confusion_matrix'] for f in metrics.keys()]
        # figure = plt.figure(figsize=(15, 15))
        # for i in range(len(val_conf_mats)):
        #     sensitivities, specificities = get_ROC_from_confusion(val_conf_mats[i])
        #     sens_0.append(sensitivities[0])
        #     spec_0.append(specificities[0])
        # sens_0.append(1)
        # spec_0.append(0)
        # plt.plot(1-np.array(spec_0), sens_0, label='Aberrant')
        # plt.xlabel('1 - Specificity', fontsize=25)
        # plt.ylabel('Sensitivity', fontsize=25)
        # plt.title(f"{name} ROC Curve", fontsize=30)
        # plt.tight_layout()
        # plt.savefig(f"{dir}/{folder}/{val_res}_ROC_curve.pdf")


    val_means_df = pd.DataFrame(val_means)
    val_stds_df = pd.DataFrame(val_stds)
    # for col in val_means_df:
    #     if 'weighted' in col:
    #         continue
    #     for i in range(len(val_means_df[col])):
    #         val_means_df[col].iloc[i] = f"{val_means_df[col].iloc[i]:.2f} +- {val_stds_df[col].iloc[i]:.2f}"


    val_means_df.index = names
    # val_means_df = val_means_df.assign(val_confusion_matrix = [[conf] for conf in mean_conf_mats_val])
    conf_mat_dict = {name: mean_conf_mat for name, mean_conf_mat in zip(names, mean_conf_mats_val)}
    with open(f"{dir}/all_experiments_{val_res}.txt", "w") as f:
        f.write(val_means_df.to_latex(formatters={"name": str.upper},
                  float_format="{:.2f}".format,))
    
    test_means_df = pd.DataFrame(test_means)
    test_means_df.index = names

    with open(f"{dir}/all_experiments_test.txt", "w") as f:
        f.write(test_means_df.to_latex(formatters={"name": str.upper},
                  float_format="{:.2f}".format,))



    
if __name__ == "__main__":
    # inspect("output/TimeFormer")
    # inspect("output/SMNN")
    # inspect("output/TimeConvNet")
    # inspect("output/GCN_TimeFormer")
    # inspect("output/noAbnormal")
    # inspect("output/3class")
    # inspect("output/best_2class")
    inspect("output/output_probs_wLabels_2class")

    # folder1 = "output/TimeFormer/FM_cls_combo_TimeFormer"
    # folder2 = "output/SMNN/FM_cls_handsfeethips_SMNN"
    # statistical_significance(folder1, folder2, mode='2class_wAbnormal')
    # folder2 = "output/TimeConvNet/FM_cls_posonly_TimeConvNet"
    # statistical_significance(folder1, folder2, mode='2class_wAbnormal')
    # folder1 = "output/SMNN/FM_cls_handsfeethips_SMNN"
    # statistical_significance(folder1, folder2, mode='2class_wAbnormal')

    # folder1 = "output/noAbnormal/FM_cls_best_combo_noAbnormal_TimeFormer"
    # folder2 = "output/noAbnormal/FM_cls_best_handsfeethips_noAbnormal_SMNN"
    # statistical_significance(folder1, folder2, mode='2class_noAbnormal')
    # folder2 = "output/noAbnormal/FM_cls_best_posonly_noAbnormal_TimeConvNet"
    # statistical_significance(folder1, folder2, mode='2class_noAbnormal')
    # folder1 = "output/noAbnormal/FM_cls_best_handsfeethips_noAbnormal_SMNN"
    # statistical_significance(folder1, folder2, mode='2class_noAbnormal')

    # folder1 = "output/3class/FM_cls_best_combo_3class_TimeFormer"
    # folder2 = "output/3class/FM_cls_handsfeethips_3class_SMNN"
    # statistical_significance(folder1, folder2, mode='3class')
    # folder2 = "output/3class/FM_cls_posonly_3class_TimeConvNet"
    # statistical_significance(folder1, folder2, mode='3class')
    # folder1 = "output/3class/FM_cls_handsfeethips_3class_SMNN"
    # statistical_significance(folder1, folder2, mode='3class')
    # folder1 = "output/SMNN/FM_cls_minAugment_SMNN"
    # folder2 = "output/SMNN/FM_cls_noAugment_SMNN"
    # statistical_significance(folder1, folder2, mode='2class_wAbnormal')


