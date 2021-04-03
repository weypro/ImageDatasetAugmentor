# 图像数据集增强器 ImageDatasetAugmentor

## 说明
该项目为图像数据集增强器，用来将图像数据集作翻转、旋转、平移、放缩、裁剪、随机噪声等操作。需要安装opencv。  
与其他图像增强项目的不同点在于，在代码编写方式上与pytorch大致相同，即将任意图像操作模块进行组合，形成一个指令序列。然后输入三维图像数组，得到最终输出。

```python
            process_module1 = Sequential(Resize(564, 360),
                            RandomRotate(10, 10)
                            )

            process_module2 = Sequential(Blur(),
                            SaltPepperNoise(0.001),
                            GaussianNoise(0,10),
                            Shift(10, 20),
                            RectClipping(10,10,200,200)
                            )


            processed_img1 = process_module1(img)
            processed_img2 = process_module2(processed_img1)

```

这样带来的好处是，对于同一张图像，可以用不同的增强方法生成多张图像，而且可以分步增强并分开保存。增强方法甚至可以随机组合。  
目前项目还处于初级阶段，未封装成独立完整的库，只能作为模板工程使用。

## 未来目标
- 增加更多的增强方法，比如变亮、变暗、对比度、平均划分、旋转后去黑边等
- 增加数据集管理，比如显示各类样本数量，分布
- 支持生成更多的数据集标签格式，比如一般的csv，到规范的coco，voc
- 增加掩膜图像的同步增强支持

欢迎大家来一起交流

