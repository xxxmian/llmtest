文件说明：

数据（datas文件夹说明）
1. ALLData是总题库，包含目前收集整理的所有数据，每个文件名都标明了数据量。
2. datas0327是此次进行测试时从题库中随机抽取的测试题。
3. test_ori.json是将data0327中题库进行合并的测试数据，方便测试执行。
4. test_sent.json是发送给用户的去答案版本的测试数据。
5. test_dev.json 是对齐数据，发送给用户，题量少，有参考答案。
6. test_responsed.json 是用户返回的数据，包含提问和模型响应。

工具（scripts文件夹说明）
1. process.py
	1）随机抽题。
	2）制作ori、sent 和dev版本的数据。
2. infer.py 是执行推理的脚本，传入模型定义即可运行、也可以发送给用户执行。
3.model_library.py 是模型拉起的配置文件。
	目前包含45个模型的推理脚本，后续可持续增加，可通过
		from model_library import get_model
		model = get_model(modelname)
	使用，modelname请参考get_model()函数中的定义。
4. judger.py 做两件事：
	1）合并用户返回的responsed文件和ori文件到responsed文件中。
	2）执行判断。
5. viser.py 包含数据统计，数据可视化的脚本。可画柱状图、雷达图，旭日图、柱状图和折线图的组合图，等。
	1. 数据处理、结果统计、总图
	2. 分模型可视化
	3. 分维度可视化
	4. 模型对比可视化
	5. 总维度可视化
