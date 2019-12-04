import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
pd.options.mode.chained_assignment = None  # default='warn'

from subprocess import check_output
print(check_output(["ls", "../dataset"]).decode("utf8"))

# order_products_train_df = pd.read_csv("../dataset/order_products__train.csv")
# order_products_prior_df = pd.read_csv("../dataset/order_products__prior.csv")
# orders_df = pd.read_csv("../dataset/orders.csv")
# products_df = pd.read_csv("../dataset/products.csv")
# aisles_df = pd.read_csv("../dataset/aisles.csv")
# departments_df = pd.read_csv("../dataset/departments.csv")