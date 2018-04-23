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
<<<<<<< HEAD


##2_SignComp文件夹结构<br>
| ---- 0_Data 存放数据<br>
|	| ---- font 存放生成文字所需的字体库<br>
|		| ---- fangzheng_fangsong.ttf<br>
|		| ---- simfang.ttf<br>
|		| ---- ...<br>
|	| ---- words 存放通过字库生成的单个汉字<br>
|		| ---- 0 编号为0的汉字<br>
|			| ---- 0_NotoSansHans-DemiLight.otf.jpg<br>
|			| ---- 1_NotoSansHans-Light.otf.jpg<br>
|			| ---- ...<br>
|			| ---- 5_simsun.ttc.jpg<br>
|		| ---- ...<br>
|		| ---- N 编号为N的汉字<br>
|	| ---- train 存放训练样本<br>
|		| ---- signs 手写签名样本<br>
|			| ---- 13410856933020180131102510091_姚创洲_2.jpg<br>
|			| ---- ...<br>
|			| ---- 13411900203020180115142056017_陈敬辉_2.jpg<br>
|		| ---- names 生成的系统文本<br>
|			| ---- 0_0.tif<br>
|			| ---- ...<br>
|			| ---- 52_1.tif<br>
|	| ---- test 存放测试样本<br>
|		| ---- signs 手写签名样本<br>
|			| ---- 13410856933020180131102510091_姚创洲_2.jpg<br>
|			| ---- ...<br>
|			| ---- 13411900203020180115142056017_陈敬辉_2.jpg<br>
|		| ---- names 生成的系统文本<br>
|			| ---- 0_0.tif<br>
|			| ---- ...<br>
|			| ---- 52_1.tif<br>
|	| ---- valid 存放模型的输出结果进行验证<br>
|		| ---- 0 姓名与系统文本一致，且预测准确<br>
|			| ---- img1_6.605199_66%_0.jpg 第二个参数表示距离<br>
|			| ---- img3_7.191527_64%_0.jpg<br>
|		| ---- 1 姓名与系统文本不一致，且预测准确<br>
|		| ---- 2 姓名与系统文本一致，且预测错误<br>
|		| ---- 3 姓名与系统文本不一致，且预测错误<br>
| ---- 1_Preprocessing 图片预处理<br>
|	| ---- chi_sim.py 存储三千多个常用的汉字，由make_ocr_dataset.py调用<br>
|	| ---- cv2_img_proc.py 自定义图像处理函数，包括切边和按比例缩放等，由lang_aux.py调用<br>
|	| ---- lang_aux.py 生成汉字的图片，由make_ocr_dataset.py调用<br>
|	| ---- utils.py 自定义函数，包括修剪文本等，由lang_aux.py调用<br>
|	| ---- make_ocr_dataset.py 从汉字生成图片的主程序<br>
| ---- 2_Generate 生成数据<br>
|	| ---- generate_name.py 根据客户的姓名生成系统文本<br>
| ---- 3_Train 模型训练<br>
|	| ---- data_Loader_siamese.py 读取数据，生成训练样本的batch，由run.py调用<br>
|	| ---- inference.py siamese模型的描述文件以及loss函数的定义<br>
|	| ---- resize.py 对手写签名进行截取，去除白边，由data_loader_siamese.py调用<br>
|	| ---- run.py 训练模型的主函数<br>
| ---- 4_Validate 模型验证<br>
|	| ---- test_sia_conv_sign.py 输出模型的精度和混淆矩阵，并且将测试样本按照预测结果分为四个文件夹<br>
| ---- 5_Eval 模型评估<br>
| ---- 6_Model 模型权重<br>
<br>
<br>
1、运行1_Preprocessing/make_ocr_dataset.py，传入参数，将常用汉字生成对应的图片，例如python make_ocr_dataset.py --output_dir imgages --font_dir font --width 64 --height 64 --margin 4 --langs chi_sim<br>
2、修改路径参数，运行两次2_Generate/make_ocr_dataset.py，分别根据训练和测试样本的签名生成对应的文本图片<br>
3、运行3_Train/run.py，进行模型的训练<br>
4、运行4_Validate/test_sia_conv_sign.py，输出模型的精度和混淆矩阵，并且将测试样本按照预测结果分为四个文件夹
=======
>>>>>>> e0d0d098c28a5ec15aed2ea2caef6f1624d56114
