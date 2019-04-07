#Code

#### 组织结构
* package: 
    * HotSpot: HotSpot算法实现
    * utils: 本项目所用到的数据结构或工具类
* prehandle: 数据预处理与数据勘探处理
* dataTransform: 数据转化，将源数据转化为utils中定义的数据结构——KPISet、KPIPoint
* KPIPredict：KPI叶子元素预测算法实现
* localization: 异常根因定位算法实现
> 某些算法实现分为.py 和 .ipynb两种文件格式。用户可根据实际选择实现的版本。