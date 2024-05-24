import wandb

api = wandb.Api()

def getResults(project_name):
    metrics = {}
    runs = api.runs(f"m46nu5/{project_name}")
    for test_fold in range(8):
        metrics[f"test_fold_{test_fold}"] = {}
        for val_fold in range(7):
            run = runs[test_fold*7 + val_fold]
            if run.state == "finished":
                summary = run.summary
                metrics[f"test_fold_{test_fold}"][f"val_fold_{val_fold}"]['val_loss'] = summary['val_loss']
                metrics[f"test_fold_{test_fold}"][f"val_fold_{val_fold}"]['val_acc'] = summary['val_acc']
                
    for run in runs:
        if run.state == "finished":
            metrics = run.summary
            # history = run.history()

            break
    else:
        print(f"Project: {project_name} not finished.")
        return




def filter_and_sort_projects():
    projects = api.projects("m46nu5")
    projects_sorted = {'TimeFormer':[], 'TimeFormer_GCN':[], 'SMNN':[], 'TimeConvNet':[]}
    for project in projects:
        for key in projects_sorted.keys():
            if key == 'TimeFormer_GCN':
                if "GCN" in project.name:
                    projects_sorted[key].append(project.name)
            else:
                if key == project.name.split('_')[-1]:
                    if key == 'TimeFormer':
                        if "final" not in project.name:
                            continue
                    projects_sorted[key].append(project.name)

    return projects_sorted


if __name__ == "__main__":
    projects_sorted = filter_and_sort_projects()

    for key in projects_sorted.keys():
        print(f"Getting results for: {key}({len(projects_sorted[key])} experiments)...")
        for project_name in projects_sorted[key]:
            getResults(project_name)

    print(f"TimeFormer: {len(projects_sorted['TimeFormer'])}")
    print(f"TimeFormer_GCN: {len(projects_sorted['TimeFormer_GCN'])}")
    print(f"SMNN: {len(projects_sorted['SMNN'])}")
    print(f"TimeConvNet: {len(projects_sorted['TimeConvNet'])}")