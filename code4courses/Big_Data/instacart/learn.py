import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
pd.options.mode.chained_assignment = None  # default='warn'

from subprocess import check_output
print(check_output(["ls", "../dataset"]).decode("utf8"))

order_products_train_df = pd.read_csv("../dataset/order_products__train.csv")
order_products_prior_df = pd.read_csv("../dataset/order_products__prior.csv")
orders_df = pd.read_csv("../dataset/orders.csv")
products_df = pd.read_csv("../dataset/products.csv")
aisles_df = pd.read_csv("../dataset/aisles.csv")
departments_df = pd.read_csv("../dataset/departments.csv")

cnt_srs = orders_df.eval_set.value_counts()

plt.figure(figsize=(12,8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Eval set type', fontsize=12)
plt.title('Count of rows in each dataset', fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


def get_unique_count(x):
    return len(np.unique(x))

cnt_srs = orders_df.groupby("eval_set")["user_id"].aggregate(get_unique_count)
print(cnt_srs)