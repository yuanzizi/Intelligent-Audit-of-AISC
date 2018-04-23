一、文件夹结构
|
| ---- 0_Data 存放数据
|	| ---- font 存放生成文字所需的字体库
|		| ---- fangzheng_fangsong.ttf
|		| ---- simfang.ttf
|		| ---- ...
|	| ---- words 存放通过字库生成的单个汉字
|		| ---- 0 编号为0的汉字
|			| ---- 0_NotoSansHans-DemiLight.otf.jpg
|			| ---- 1_NotoSansHans-Light.otf.jpg
|			| ---- ...
|			| ---- 5_simsun.ttc.jpg
|		| ---- ...
|		| ---- N 编号为N的汉字
|	| ---- train 存放训练样本
|		| ---- signs 手写签名样本
|			| ---- 13410856933020180131102510091_姚创洲_2.jpg
|			| ---- ...
|			| ---- 13411900203020180115142056017_陈敬辉_2.jpg
|		| ---- names 生成的系统文本
|			| ---- 0_0.tif
|			| ---- ...
|			| ---- 52_1.tif
|	| ---- test 存放测试样本
|		| ---- signs 手写签名样本
|			| ---- 13410856933020180131102510091_姚创洲_2.jpg
|			| ---- ...
|			| ---- 13411900203020180115142056017_陈敬辉_2.jpg
|		| ---- names 生成的系统文本
|			| ---- 0_0.tif
|			| ---- ...
|			| ---- 52_1.tif
|	| ---- valid 存放模型的输出结果进行验证
|		| ---- 0 姓名与系统文本一致，且预测准确
|			| ---- img1_6.605199_66%_0.jpg 第二个参数表示距离
|			| ---- img3_7.191527_64%_0.jpg
|		| ---- 1 姓名与系统文本不一致，且预测准确
|		| ---- 2 姓名与系统文本一致，且预测错误
|		| ---- 3 姓名与系统文本不一致，且预测错误
| ---- 1_Preprocessing 图片预处理
|	| ---- chi_sim.py 存储三千多个常用的汉字，由make_ocr_dataset.py调用
|	| ---- cv2_img_proc.py 自定义图像处理函数，包括切边和按比例缩放等，由lang_aux.py调用
|	| ---- lang_aux.py 生成汉字的图片，由make_ocr_dataset.py调用
|	| ---- utils.py 自定义函数，包括修剪文本等，由lang_aux.py调用
|	| ---- make_ocr_dataset.py 从汉字生成图片的主程序
| ---- 2_Generate 生成数据
|	| ---- generate_name.py 根据客户的姓名生成系统文本
| ---- 3_Train 模型训练
|	| ---- data_Loader_siamese.py 读取数据，生成训练样本的batch，由run.py调用
|	| ---- inference.py siamese模型的描述文件以及loss函数的定义
|	| ---- resize.py 对手写签名进行截取，去除白边，由data_loader_siamese.py调用
|	| ---- run.py 训练模型的主函数
| ---- 4_Validate 模型验证
|	| ---- test_sia_conv_sign.py 输出模型的精度和混淆矩阵，并且将测试样本按照预测结果分为四个文件夹
| ---- 5_Eval 模型评估
| ---- 6_Model 模型权重



二、使用方法
1、运行1_Preprocessing/make_ocr_dataset.py，传入参数，将常用汉字生成对应的图片，例如python make_ocr_dataset.py --output_dir imgages --font_dir font --width 64 --height 64 --margin 4 --langs chi_sim
2、修改路径参数，运行两次2_Generate/make_ocr_dataset.py，分别根据训练和测试样本的签名生成对应的文本图片
3、运行3_Train/run.py，进行模型的训练
4、运行4_Validate/test_sia_conv_sign.py，输出模型的精度和混淆矩阵，并且将测试样本按照预测结果分为四个文件夹