一、文件夹结构
|
| ---- 0_Data 存放数据
|	| ---- train 存放训练样本
|		| ---- 0 存放类别0的样本
|		| ---- 1 存放类别1的样本
|		| ---- ...
|		| ---- 9 存放类别9的样本
|	| ---- test 存放测试样本
|		| ---- 0 存放类别0的样本
|		| ---- 1 存放类别1的样本
|		| ---- ...
|		| ---- 9 存放类别9的样本
|	| ---- output 存放模型输出的数据及图片
| ---- 1_Preprocessing 图片预处理
|	| ---- crop_white.py 切除pdf生成的图片中白块部分
|	| ---- filter_watermark.py 将有pdf生成的图片中，有水印的图片过滤掉
|	| ---- gray2rgb.py 将灰度图转成RGB三通道图
| ---- 2_Generate 生成数据
|	| ---- tfr_generator.py 将图片生成tensorflow训练和测试用的二进制文件
|	| ---- tfr_generator_300.py 将图片生成tensorflow训练和测试用的二进制文件（图片像素为300*300）
| ---- 3_Train 模型训练
|	| ---- train.py 训练文件
|	| ---- model1_0315.py 深度学习模型的结构描述文件（分为9类，最终采用）
|	| ---- model1_0320.py 深度学习模型的结构描述文件（分为8类）
|	| ---- model1_300.py 深度学习模型的结构描述文件（图片像素为300*300）
| ---- 4_Validate 模型验证
|	| ---- img_class.py 根据模型对工单图片进行分类
|	| ---- model_metrics.py 输出模型的相关指标，如准确率、覆盖率和混淆矩阵
|	| ---- save_score.py 输出每个训练样本对应的9个类别的概率
|	| ---- save_score_300.py 输出每个训练样本对应的9个类别的概率（图片像素为300*300）
|	| ---- model1_0315.py 深度学习模型的结构描述文件（分为9类，最终采用）
|	| ---- model1_0320.py 深度学习模型的结构描述文件（分为8类）
|	| ---- model1_300.py 深度学习模型的结构描述文件（图片像素为300*300）
| ---- 5_Model 模型权重
|	| ---- models 保留训练完成后的模型权重


二、使用方法
1、运行tfr_generator.py，生成相应的训练和测试样本
2、运行train.py，对模型进行训练
3、运行model_metrics.py，观察模型的相关指标
4、运行img_class.py，对已有的工单进行分类