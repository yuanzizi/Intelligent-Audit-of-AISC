一、文件夹结构
|
| ---- 0_Data 存放数据
|	| ---- rotation 旋转图片的相关数据
|		| ---- train 存放训练样本
|			| ---- 0 存放正面向左的样本
|			| ---- 1 存放正面朝上的样本
|			| ---- 2 存放正面朝右的样本
|			| ---- 3 存放正面朝下的样本
|			| ---- 4 存放免密保证书电子版的样本
|		| ---- test 存放测试样本
|			| ---- 0 存放正面向左的样本
|			| ---- 1 存放正面朝上的样本
|			| ---- 2 存放正面朝右的样本
|			| ---- 3 存放正面朝下的样本
|			| ---- 4 存放免密保证书电子版的样本
|		| ---- output 存放模型输出的数据及图片
|	| ---- sign2 补卡保证书的相关数据
|		| ---- Annotations 存放标注信息
|			| ---- 000000.xml
|			| ---- 000001.xml
|			| ---- ...
|			| ---- 00000N.xml
|		| ---- ImageSets 存放训练/验证/测试集合的指示文档
|				| ---- Main 
|					| ---- train.txt 存放用来训练的图片的序号
|					| ---- val.txt 存放用来验证的图片的序号
|					| ---- trainval.txt 存放用来训练/验证的图片的序号
|					| ---- test.txt 存放用来测试的图片的序号
|		| ---- JPEGImages 存放相关的所有图片
|		| ---- mAP 存放评估模型精度的相关文件
|			| ---- ground-truth 存放验证集合的实际结果
|			| ---- predicted 存放验证集合的预测结果
|			| ---- images 存放验证集合的图片（也可以直接将JPEGImages软链接过来）
|			| ---- results 存放模型精度的评估结果
|		| ---- 7_label_map.pbtxt 描述需要标注的物体类型和ID的文件
|		| ---- sign2_trainval.record 训练和验证目标检测模型的二进制文件
|		| ---- sign2_test.record 测试目标检测模型的二进制文件
|	| ---- sign3 免密保证书的相关数据（文档结构与sign2一样）
| ---- 1_Preprocessing 图片预处理
|	| ---- crop2label.py 针对补卡保证书/免密保证书进行局部放大的脚本（测试用）
|	| ---- rotate_2train.py 批量对图片进行旋转，方便生成训练样本
|	| ---- rotate_img4label.py 检测工单的类型和方向，将图片旋转到正面朝上并局部放大
|	| ---- generage_img4label.py 识别工单类型并分类进行保存
|	| ---- filter_img2label.py 分别从两种保证书的文件夹中，将概率低于阈值的图片删除掉（净化训练样本）
|	| ---- model1_0315.py 工单分类模型的结构描述文件
| ---- 2_Generate 生成数据
|	| ---- tfr_generator_direction.py 将图片生成训练和测试旋转图片模型用的二进制文件
|	| ---- txt_generate.py 将图片集合分为训练/验证/测试集合并生成相应的记录文档
|	| ---- run.txt 运行文档里面的代码，生成目标检测用的训练和测试用的二进制文件
| ---- 3_Train 模型训练
|	| ---- train_direction.py 训练图片旋转模型的脚本
|	| ---- model3_0315.py 图片旋转模型的结构描述文件
|	| ---- run.txt 运行文档里面的代码，训练目标检测模型，并将模型的参数和输出在tensorboard上进行展示
| ---- 4_Validate 模型验证
| ---- 5_Eval 模型验证
|	| ---- sign2_convert_gt_xml.py 将sign2测试集合的标注文件转成txt文件
|	| ---- sign2_output_pos.py 将模型在sign2测试集合上的预测结果转成txt文件
|	| ---- sign2_main.py 评估sign2模型在mAP上的得分和具体每张图的结果
|	| ---- sign3_convert_gt_xml.py 将sign3测试集合的标注文件转成txt文件
|	| ---- sign3_output_pos.py 将模型在sign3测试集合上的预测结果转成txt文件
|	| ---- sign3_main.py 评估sign3模型在mAP上的得分和具体每张图的结果
| ---- 6_Model 模型权重
|	| ---- rotation 保留训练完成后的图片旋转模型的权重
|	| ---- sign2 保留训练完成后的sign2模型的权重
|	| ---- sign3 保留训练完成后的sign3模型的权重



二、使用方法
1、训练图片旋转模型
	（1）运行1_Preprocessing/generage_img4label.py，将原始工单图片进行分类，并保存到不同的文件夹
	（2）运行1_Preprocessing/filter_img2label.py，分别从两种保证书的文件夹中剔除掉概率低于阈值的图片
	（3）将两种保证书的工单放到一起，并手动旋转到正面朝上的方向，然后运行1_Preprocessing/rotate_2train.py，批量对图片进行旋转并保存到不同的文件夹
	（4）运行2_Generate/tfr_generator_direction.py，将图片生成训练和测试旋转图片模型用的二进制文件
	（5）运行3_Train/train_direction.py，训练图片旋转模型
2、安装tensorflow object detection APIS（详见install_tensorflow_object _detection_API.txt）
3、训练sign2目标检测模型（sign3的执行步骤类似）
	（1）运行2_Generate\txt_generate.py 将图片集合分为训练/验证/测试集合并生成相应的记录文档
	（2）运行2_Generate\run.txt文档里面的代码，生成目标检测用的训练和测试用的二进制文件
	（3）运行3_Train\run.txt文档里面的代码，训练目标检测模型，并将模型的参数和输出在tensorboard上进行展示
	（3）运行5_Eval\sign2_convert_gt_xml.py，将sign2测试集合的标注文件转成txt文件
	（4）运行5_Eval\sign2_output_pos.py，将模型在sign2测试集合上的预测结果转成txt文件
	（5）运行5_Eval\sign2_main.py，评估sign2模型在mAP上的得分和具体每张图的结果