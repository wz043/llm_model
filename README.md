                                                            GMT_Zero
这个模型是个人搭建和训练的一个语言模型，训练成本将近900块钱，模型架构和权重都已公开，预训练采用了序列猴子数据集魔搭社区有，后训练用的belle_data1M_cn.json数据集，然后训练曲线公开在https://swanlab.cn/@jwz012/GMT_alpha_0.6B/overview
，其中的jiangxin训练暂未查明loss激增原因，最终模型只经过per_train和post_train，
蒸馏损失代码实际并没有被应用但是这里提供了一个简单模板，模型参数将近0.6B，主体架构采用的GQA和门控MLP，
模型最终效果：
![IMG_20251103_212306](https://github.com/user-attachments/assets/1b1d6329-2d04-405b-ba2d-6fdf241fdcf6)
