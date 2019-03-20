## 文件组织

- code
	- **anomaly injection.ipynb** : 异常注入源码；

- data
	- **origin** ： 原始KPI数据集，按时间戳划分；
	- **result** ： 按第layer 1 ；属性值划分数据集，每个属性每个值对应的KPI时间序列；
	- **anomaly** ： 异常注入源码运行后————在随机timestamp上，为随机生成的25种根因注入异常，将总异常值平均分发到此其在timestamp上的所有后代叶子元素中；
	- **KPI_new.csv** ： 每个时间戳的总KPI;
	- **root_causes.npy** ： 异常注入源码运行后————随机生成的25种根因数组保存成文件，便于定位根因的结果评估。