import pandas as pd
import numpy as np
import scipy.stats as ss
from src.utils.eval_util import cal_metric
lines = []
for i in range(4):
    lines += open('./data/L/result/split_{}.txt'.format(i), 'r').readlines()

group_preds = {}
group_labels = {}

for l in lines:
    row = l.strip().split('\t')
    if row[0] not in group_preds:
        group_preds[row[0]] = []
    if row[0] not in group_labels:
        group_labels[row[0]] = []
    group_preds[row[0]].append(float(row[-1]))
    group_labels[row[0]].append(int(row[1]))

all_labels = []
all_preds = []
for k in all_keys:
    all_labels.append(group_labels[k])
    all_preds.append(group_preds[k])

metric_list = [x.strip() for x in "group_auc || mean_mrr || ndcg@5;10".split("||")]
ret = cal_metric(all_labels, all_preds, metric_list)
for metric, val in ret.items():
    print("Epoch: {}, {}: {}".format(1, metric, val))

# with open('prediction.txt', 'w') as fw:
#     for k in group_preds:
#         rank = ss.rankdata(-np.array(group_preds[k])).astype(int).tolist()
#         rank_str = '[' + ','.join(list(map(str, rank))) + ']'
#         fw.write("{} {}\n".format(k, rank_str))

# df = pd.read_csv('prediction.txt', sep=' ', names=['impid','rk'])
# df = df.sort_values(by='impid', ascending=True)
# df.to_csv('prediction.txt', sep=' ', index=0, index_label=0, header=0)
