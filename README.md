# Accurate-localization-of-key-points-Scleral-Spur-based-on-VGG19-on-ACA-and-measurement-of-relevant
In previous studies, classification tasks were completed, and the ACA was divided into three categories: open, closed, and narrow. However, it is important to locate the scleral-spur of open angle and narrow angle  and measure the relevant parameters.  
在之前的研究中，完成了分类任务，并将ACA分为三类：开、闭和窄三类。然而，对开角和窄角的巩膜突的定位及相关参数的测量是非常重要的。下面两幅图为医院医生标定的结果。    
![image](https://github.com/1579477793/Accurate-localization-of-key-points-Scleral-Spur-based-on-VGG19-on-ACA-and-measurement-of-relevant/blob/master/results/sample1.png)  
![image](https://github.com/1579477793/Accurate-localization-of-key-points-Scleral-Spur-based-on-VGG19-on-ACA-and-measurement-of-relevant/blob/master/results/sample2.png)  
save_data.py 是对原始图像进行处理的程序，图像转换为.npy形式。 
train_dropou.py 用VGG19对数据进行训练  
数据一共有四个：  
训练集：图像x_train.npy, 标签y_train.npy  
测试集：图像x_test.npy, 标签y_test.npy  
实验结果如下：  
![image](https://github.com/1579477793/Accurate-localization-of-key-points-Scleral-Spur-based-on-VGG19-on-ACA-and-measurement-of-relevant/blob/master/results/1.png)
![image](https://github.com/1579477793/Accurate-localization-of-key-points-Scleral-Spur-based-on-VGG19-on-ACA-and-measurement-of-relevant/blob/master/results/2.png)
![image](https://github.com/1579477793/Accurate-localization-of-key-points-Scleral-Spur-based-on-VGG19-on-ACA-and-measurement-of-relevant/blob/master/results/3.png)
![image](https://github.com/1579477793/Accurate-localization-of-key-points-Scleral-Spur-based-on-VGG19-on-ACA-and-measurement-of-relevant/blob/master/results/4.png)  
目前卷积的总体框架基本完成，下面根据巩膜突（scleral spur）的位置，进行一系列的参数标定与测量，设计边缘检测，直线拟合，确定垂线，计算交点等过程。  
![image](https://github.com/1579477793/Accurate-localization-of-key-points-Scleral-Spur-based-on-VGG19-on-ACA-and-measurement-of-relevant/blob/master/results/6.png)
