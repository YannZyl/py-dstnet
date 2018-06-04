# py-dstnet

基于深度时空网络的异常操作识别算法.


# 实验环境

- [tensorflow](https://www.tensorflow.org/)
- [keras](https://keras.io/)
- opencv

# 数据采集说明

算法如果识别N=4类操作，那么需要在datas文件夹下面创建4个子文件夹，分别为：

- datas/videos/op1
- datas/videos/op2
- datas/videos/op3
- datas/videos/op4

然后对每个行为进行视频采集，放入对应的文件夹即可。

参考数据集：[KTH](http://www.nada.kth.se/cvap/actions/)

# 训练

```python
python model.py
```
