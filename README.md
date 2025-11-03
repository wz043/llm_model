                                                        GMT_Zero
这个模型是个人搭建和训练的一个语言模型，模型参数将近0.6B，主体架构采用的GQA和门控MLP，训练成本将近900块钱，显卡pro6000单卡，有效训练总时间将近2天（预训练和后训练除去试错时间），模型架构和权重都已公开，预训练采用了序列猴子数据集魔搭社区有，后训练用的belle_data1M_cn.json数据集，然后训练曲线公开在https://swanlab.cn/@jwz012/GMT_alpha_0.6B/overview
，其中的匠心数据集训练时loss激增原因暂未查明，最终模型只经过per_train和post_train，
文件SFT_tool是一个数据处理函数可以用在预训练和后训练上面,distill_tool里面是蒸馏损失，蒸馏损失代码实际并没有被应用但是这里提供了一个简单模板
模型最终效果：
![IMG_20251103_212306](https://github.com/user-attachments/assets/1b1d6329-2d04-405b-ba2d-6fdf241fdcf6)
