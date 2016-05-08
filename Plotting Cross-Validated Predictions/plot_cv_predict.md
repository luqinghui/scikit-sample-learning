# Plotting Cross-Validated Predictions
---
该实例展示了如何使用cross\_val\_predict函数可视化预测误差

![pic](http://scikit-learn.org/stable/_images/plot_cv_predict_001.png)

**python源码：**

	from sklearn import datasets
	from sklearn.cross_validation import cross_val_predict
	from sklearn import linear_model
	import matplotlib.pyplot as plt
	
	lr = linear_model.LinearRegression()
	boston = datasets.load_boston()
	y = boston.target
	
	# cross_val_predict returns an array of the same size as `y` where each entry
	# is a prediction obtained by cross validated:
	predicted = cross_val_predict(lr, boston.data, y, cv=10)
	
	fig, ax = plt.subplots()
	ax.scatter(y, predicted)
	ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	plt.show()


函数讲解：

> ```python
> sklearn.cross_validation.cross_val_predict(estimator, X, y=None, cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')
> ```
>
> 参数说明：
> 
> **cv:** 
>
> 输入是整数或None，如果y是二分类或多类，采用StratifiedKFold。如果是estimator是分类器，但y不是二类或多类，采用KFold。
>
> 使用K折检验，默认3折，最少为2折
> 
> 1. **StratifiedKFold:**k折数据按照百分比划分数据集，每个类别百分比在训练集和测试集中都是一样。
> 2. **KFold:**普通
> 3. **LabelKFold：**某个样例的数据必须是属于训练集或者测试集时，可用这个函数。


##参考文献：
1. [python sklearn 3.1cross validation笔记][1]
2. [StratifiedKFold][2]
3. [KFold][3]
4. [LabelFold][4]
5. [cross\\_val\\_predict][5]

[1]: http://blog.csdn.net/u010454729/article/details/50754076
[2]: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedKFold.html#sklearn.cross_validation.StratifiedKFold
[3]: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html#sklearn.cross_validation.KFold
[4]: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.LabelKFold.html#sklearn.cross_validation.LabelKFold
[5]: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_predict.html#sklearn.cross_validation.cross_val_predict



