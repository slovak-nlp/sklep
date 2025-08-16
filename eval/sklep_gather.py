import os
import sys
import json
import polars as pl


# for each task, select the representative argument and its reference value
# for now reference values are from the paper
all_tasks = {
        "qa": ("eval_f1", 0.74356969),
        "sts": ("eval_pearson", 0.831849),
        "nli": ("eval_accuracy", 0.827538),
        "rte": ("eval_accuracy", 0.652008),
        "hate": ("eval_accuracy", 0.803134),
        "sentiment": ("eval_accuracy", 0.976328),
        "uner":("eval_f1", 0.819164),
        "wikigold": ("eval_f1", 0.921420),
        "pos":("eval_f1", 0.980414),
}

outdir = os.path.normpath(sys.argv[1])

def safe_select(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    selected_cols = []
    for col in columns:
        if col in df.columns:
            selected_cols.append(pl.col(col))
        else:
            selected_cols.append(pl.lit(None).alias(col))
    return df.select(selected_cols)

def print_md(df):
    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
    ):
        print(df)

model_names = set()
raw_results = []
final_results = []
for dirpath, dirnames, filenames in os.walk(outdir):
   for filename in filenames:
      if filename == "eval_results.json":
         with open(dirpath + "/" + filename) as f:
             items = dirpath.split("/")
             task_name = items[1]
             model_name = items[2]
             run_name = items[3]
             doc = json.load(f)
             model_names.add(model_name)
             for k,v in doc.items():
                 # gather all results from the run
                 raw_results.append({
                     "model_name":model_name,
                     "task_name":task_name,
                     "run_name":run_name,
                     "metric":k,
                     "value": v 
                 })
                 # pick the final result from the run
                 if task_name in all_tasks and all_tasks[task_name][0] == k:
                     # in question answering f1 is in percent
                     # have to be compensated
                     if task_name == "qa":
                         v = v / 100
                     final_results.append({
                         "model_name":model_name,
                         "task_name":task_name,
                         "run_name":run_name,
                         "value": v 
                     })
rrf = pl.DataFrame(raw_results)


frf = pl.DataFrame(final_results)
#sorted_runs = list(run_names).sort()

print("\n## Model Results \n")
# Detailed results for each model
# If there are multiple runs, arithetic average is calculated
for model in model_names:
    print("\n### " + model + "\n")
    res = rrf.filter(pl.col("model_name")==model).sort("run_name").pivot("metric",values="value",index="task_name",aggregate_function="mean")
    res.write_csv(outdir + "/" + model + ".csv" )
    with pl.Config(tbl_rows=20, float_precision=6):
        print_md(safe_select(res, ["task_name","eval_f1","eval_accuracy","eval_pearson"]))

# Results for each task
# Calculate mean from each run
# then distract the reference value to calculate reduction
# If there are multiple runs, average is calculated

print("\n## Task Results \n")
final_results = []
for task_name,(v,reference) in all_tasks.items():
    print("\n### " + task_name + "\n")
    res = frf.filter(pl.col("task_name")==task_name).sort("run_name").group_by("model_name","task_name")\
        .agg(pl.col("value").mean(),(pl.col("value").mean() - reference).alias("reduction"), pl.col("value").std().alias("std"),pl.len().alias("runs")).sort("reduction", descending=True)
    final_results.append(res)
    with pl.Config(tbl_rows=20, float_precision=6):
        print_md(res)

leader_board = pl.concat(final_results)
# Leader Board
# Calculate average value of tasks and average reduction
leader_board = leader_board.group_by("model_name")\
        .agg(pl.col("value").mean(),pl.col("reduction").mean()).sort("reduction", descending=True)
print("\n## Leader Board\n")
with pl.Config(tbl_rows=20, float_precision=6):
    print_md(leader_board)

# TODO save last raw results
# TODO final results
