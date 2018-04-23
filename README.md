# Intelligent-Audit-of-AISC<br>
##1_ImageClass文件夹结构<br>
| ---- 0_Data 存放数据<br>
|	| ---- train 存放训练样本<br>
|		| ---- 0 存放类别0的样本<br>
|		| ---- 1 存放类别1的样本<br>
|		| ---- ...<br>
|		| ---- 9 存放类别9的样本<br>
|	| ---- test 存放测试样本<br>
|		| ---- 0 存放类别0的样本<br>
|		| ---- 1 存放类别1的样本<br>
|		| ---- ...<br>
|		| ---- 9 存放类别9的样本<br>
|	| ---- output 存放模型输出的数据及图片<br>
| ---- 1_Preprocessing 图片预处理<br>
|	| ---- crop_white.py 切除pdf生成的图片中白块部分<br>
|	| ---- filter_watermark.py 将有pdf生成的图片中，有水印的图片过滤掉<br>
|	| ---- gray2rgb.py 将灰度图转成RGB三通道图<br>
| ---- 2_Generate 生成数据<br>
|	| ---- tfr_generator.py 将图片生成tensorflow训练和测试用的二进制文件<br>
|	| ---- tfr_generator_300.py 将图片生成tensorflow训练和测试用的二进制文件（图片像素为300*300）<br>
| ---- 3_Train 模型训练<br>
|	| ---- train.py 训练文件<br>
|	| ---- model1_0315.py 深度学习模型的结构描述文件（分为9类，最终采用）<br>
|	| ---- model1_0320.py 深度学习模型的结构描述文件（分为8类）<br>
|	| ---- model1_300.py 深度学习模型的结构描述文件（图片像素为300*300）<br>
| ---- 4_Validate 模型验证<br>
|	| ---- img_class.py 根据模型对工单图片进行分类<br>
|	| ---- model_metrics.py 输出模型的相关指标，如准确率、覆盖率和混淆矩阵<br>
|	| ---- save_score.py 输出每个训练样本对应的9个类别的概率<br>
|	| ---- save_score_300.py 输出每个训练样本对应的9个类别的概率（图片像素为300*300）<br>
|	| ---- model1_0315.py 深度学习模型的结构描述文件（分为9类，最终采用）<br>
|	| ---- model1_0320.py 深度学习模型的结构描述文件（分为8类）<br>
|	| ---- model1_300.py 深度学习模型的结构描述文件（图片像素为300*300）<br>
| ---- 5_Model 模型权重<br>
|	| ---- models 保留训练完成后的模型权重<br>
<br>
1、运行tfr_generator.py，生成相应的训练和测试样本<br>
2、运行train.py，对模型进行训练<br>
3、运行model_metrics.py，观察模型的相关指标<br>
4、运行img_class.py，对已有的工单进行分类<br>
<br>
<br>
##2_SignDetection文件夹结构<br>
|<br>
| ---- 0_Data 存放数据<br>
|	| ---- rotation 旋转图片的相关数据<br>
|		| ---- train 存放训练样本<br>
|			| ---- 0 存放正面向左的样本<br>
|			| ---- 1 存放正面朝上的样本<br>
|			| ---- 2 存放正面朝右的样本<br>
|			| ---- 3 存放正面朝下的样本<br>
|			| ---- 4 存放免密保证书电子版的样本<br>
|		| ---- test 存放测试样本<br>
|			| ---- 0 存放正面向左的样本<br>
|			| ---- 1 存放正面朝上的样本<br>
|			| ---- 2 存放正面朝右的样本<br>
|			| ---- 3 存放正面朝下的样本<br>
|			| ---- 4 存放免密保证书电子版的样本<br>
|		| ---- output 存放模型输出的数据及图片<br>
|	| ---- sign2 补卡保证书的相关数据<br>
|		| ---- Annotations 存放标注信息<br>
|			| ---- 000000.xml<br>
|			| ---- 000001.xml<br>
|			| ---- ...<br>
|			| ---- 00000N.xml<br>
|		| ---- ImageSets 存放训练/验证/测试集合的指示文档<br>
|				| ---- Main <br>
|					| ---- train.txt 存放用来训练的图片的序号<br>
|					| ---- val.txt 存放用来验证的图片的序号<br>
|					| ---- trainval.txt 存放用来训练/验证的图片的序号<br>
|					| ---- test.txt 存放用来测试的图片的序号<br>
|		| ---- JPEGImages 存放相关的所有图片<br>
|		| ---- mAP 存放评估模型精度的相关文件<br>
|			| ---- ground-truth 存放验证集合的实际结果<br>
|			| ---- predicted 存放验证集合的预测结果<br>
|			| ---- images 存放验证集合的图片（也可以直接将JPEGImages软链接过来）<br>
|			| ---- results 存放模型精度的评估结果<br>
|		| ---- 7_label_map.pbtxt 描述需要标注的物体类型和ID的文件<br>
|		| ---- sign2_trainval.record 训练和验证目标检测模型的二进制文件<br>
|		| ---- sign2_test.record 测试目标检测模型的二进制文件<br>
|	| ---- sign3 免密保证书的相关数据（文档结构与sign2一样）<br>
| ---- 1_Preprocessing 图片预处理<br>
|	| ---- crop2label.py 针对补卡保证书/免密保证书进行局部放大的脚本（测试用）<br>
|	| ---- rotate_2train.py 批量对图片进行旋转，方便生成训练样本<br>
|	| ---- rotate_img4label.py 检测工单的类型和方向，将图片旋转到正面朝上并局部放大<br>
|	| ---- generage_img4label.py 识别工单类型并分类进行保存<br>
|	| ---- filter_img2label.py 分别从两种保证书的文件夹中，将概率低于阈值的图片删除掉（净化训练样本）<br>
|	| ---- model1_0315.py 工单分类模型的结构描述文件<br>
| ---- 2_Generate 生成数据<br>
|	| ---- tfr_generator_direction.py 将图片生成训练和测试旋转图片模型用的二进制文件<br>
|	| ---- txt_generate.py 将图片集合分为训练/验证/测试集合并生成相应的记录文档<br>
|	| ---- run.txt 运行文档里面的代码，生成目标检测用的训练和测试用的二进制文件<br>
| ---- 3_Train 模型训练<br>
|	| ---- train_direction.py 训练图片旋转模型的脚本<br>
|	| ---- model3_0315.py 图片旋转模型的结构描述文件<br>
|	| ---- run.txt 运行文档里面的代码，训练目标检测模型，并将模型的参数和输出在tensorboard上进行展示<br>
| ---- 4_Validate 模型验证<br>
| ---- 5_Eval 模型验证<br>
|	| ---- sign2_convert_gt_xml.py 将sign2测试集合的标注文件转成txt文件<br>
|	| ---- sign2_output_pos.py 将模型在sign2测试集合上的预测结果转成txt文件<br>
|	| ---- sign2_main.py 评估sign2模型在mAP上的得分和具体每张图的结果<br>
|	| ---- sign3_convert_gt_xml.py 将sign3测试集合的标注文件转成txt文件<br>
|	| ---- sign3_output_pos.py 将模型在sign3测试集合上的预测结果转成txt文件<br>
|	| ---- sign3_main.py 评估sign3模型在mAP上的得分和具体每张图的结果<br>
| ---- 6_Model 模型权重<br>
|	| ---- rotation 保留训练完成后的图片旋转模型的权重<br>
|	| ---- sign2 保留训练完成后的sign2模型的权重<br>
|	| ---- sign3 保留训练完成后的sign3模型的权重<br>
<br>
<br>
1、训练图片旋转模型<br>
	（1）运行1_Preprocessing/generage_img4label.py，将原始工单图片进行分类，并保存到不同的文件夹<br>
	（2）运行1_Preprocessing/filter_img2label.py，分别从两种保证书的文件夹中剔除掉概率低于阈值的图片<br>
	（3）将两种保证书的工单放到一起，并手动旋转到正面朝上的方向，然后运行1_Preprocessing/rotate_2train.py，批量对图片进行旋转并保存到不同的文件夹<br>
	（4）运行2_Generate/tfr_generator_direction.py，将图片生成训练和测试旋转图片模型用的二进制文件<br>
	（5）运行3_Train/train_direction.py，训练图片旋转模型<br>
2、安装tensorflow object detection APIS（详见install_tensorflow_object _detection_API.txt）<br>
3、训练sign2目标检测模型（sign3的执行步骤类似）<br>
	（1）运行2_Generate\txt_generate.py 将图片集合分为训练/验证/测试集合并生成相应的记录文档<br>
	（2）运行2_Generate\run.txt文档里面的代码，生成目标检测用的训练和测试用的二进制文件<br>
	（3）运行3_Train\run.txt文档里面的代码，训练目标检测模型，并将模型的参数和输出在tensorboard上进行展示<br>
	（3）运行5_Eval\sign2_convert_gt_xml.py，将sign2测试集合的标注文件转成txt文件<br>
	（4）运行5_Eval\sign2_output_pos.py，将模型在sign2测试集合上的预测结果转成txt文件<br>
	（5）运行5_Eval\sign2_main.py，评估sign2模型在mAP上的得分和具体每张图的结果<br>
