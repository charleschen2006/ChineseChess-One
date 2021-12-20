import hashlib
import json
import os
from logging import getLogger

import tensorflow as tf

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed, ActionLabelsBlack

logger = getLogger(__name__)

class CChessModel:

    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: Model
        self.digest = None
        self.n_labels = len(ActionLabelsRed)
        self.graph = None   # 这一行数据科学组注释掉了
        self.api = None

    def build(self):
        mc = self.config.model
                # 1. 输入层Input   14 x 10 x 9 :  10x9 为棋盘状态, 14是所有棋子种类（红/黑算不同种类),
        # 整体的输入就是14个棋盘堆叠在一起，种类在前, 所以使用data_format=channels_first参数
        # 每个棋盘表示一种棋子的位置：棋子所在的位置为1，其余位置为0。
        in_x = x = Input((14, 10, 9)) # 14 x 10 x 9

         # (batch, channels, height, width)
        #2. 二维卷积层
        #这一层创建了一个卷积核，它与这一层的输入卷积以产生一个输出张量。
        #如果 use_bias为真，则创建一个偏差向量并添加到输出中。此处为false, 不输出:当卷积层后根由BatchNorm或者InstanceNorm层时，最好设为False，因为归一化层会归一化卷积层输出并且加上自己的bias，卷积层的（如果有）bias就是多余的了。
        #filter: int 类型，表示卷积核个数，filters=256, 基底特征数需要设置大点, 影响的是最后输入结果的的第四个维度的变化 OUTPUT:(4, 260,260,260,260,260, 256)
        #kernel_size: 表示卷积核的大小，如果是方阵可以直接写成一个数，影响的是输出结果中间两个数据的维度kernel_size=5 即为:(5,5)  OUTPUT:(4, 260,260,260,260,260, 256)
        #padding='same' 与pytorch不同，keras和TensorFlow设置卷积层的过程中可以设置padding参数，vaild和same。“valid”代表只进行有效的卷积，对边界数据不处理。
        # “same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
        #channels_last为(batch,height,width,channels),channels_first为(batch,channels,height,width),default为channels_last.
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_first_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name="input_conv-"+str(mc.cnn_first_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        #Conv2D返回值为一个四维张量：第一个数是 batch 的大小，也就是有几组数据；后三个数表示一个张量的大小

        #3. 批量正则化层: 一般来说要设成True。
        x = BatchNormalization(axis=1, name="input_batchnorm")(x)

        #4. 激活函数 使用relu线性整流函数, 将引入非线性
        x = Activation("relu", name="input_relu")(x)
        

        """
        在class ModelConfig中配置 
        self.res_layer_num = 7
        """
        # logger.debug(f"构建{self.res_layer_num}层残差网络 ~")
        for i in range(mc.res_layer_num):
            x = self._build_residual_block(x, i + 1)
        #保存残差层结果
        res_out = x

        # for policy output
        #残差网络输出结果输入到策略价值网络中  data_format ="channels_first"表示通道在前
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False, 
                    kernel_regularizer=l2(mc.l2_reg), name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        policy_out = Dense(self.n_labels, kernel_regularizer=l2(mc.l2_reg), activation="softmax", name="policy_out")(x)

        # for value output
        #残差网络输出结果输入到行动价值网络中
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False, 
                    kernel_regularizer=l2(mc.l2_reg), name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=1, name="value_batchnorm")(x)
        x = Activation("relu",name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg), activation="relu", name="value_dense")(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="value_out")(x)

        #使用输入in_x和 [policy_out, value_out] 作为输出进行网络构建
        self.model = Model(in_x, [policy_out, value_out], name="cchess_model")
        #这行数据科学组注释掉了
        self.graph = tf.get_default_graph()

    #构建残差网络（resnet）， 处理深度网络中梯度消失问题
    def _build_residual_block(self, x, index):
        mc = self.config.model
        in_x = x
        res_name = "res" + str(index)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg), 
                   name=res_name+"_conv1-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name=res_name+"_batchnorm1")(x)
        x = Activation("relu",name=res_name+"_relu1")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg), 
                   name=res_name+"_conv2-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="res"+str(index)+"_batchnorm2")(x)
        x = Add(name=res_name+"_add")([in_x, x])
        x = Activation("relu", name=res_name+"_relu2")(x)
        return x

    @staticmethod
    def fetch_digest(weight_path):
        #申明为静态方法， 作为复用函数, 生成权重文件的hash签名
        if os.path.exists(weight_path):
            m = hashlib.sha256() #使用sha256进行加密
            with open(weight_path, "rb") as f:
                print(f"model.py line118 以字节方式读取权重文件")
                m.update(f.read()) #向 update() 输入字符串对象是不被支持的，因为哈希基于字节而非字符。
            return m.hexdigest() #返回十六进制的文件加密摘要
        return None

    #加载模型文件
    def load(self, config_path, weight_path):
        if os.path.exists(config_path) and os.path.exists(weight_path):
            logger.debug(f"model.py line126 loading model from {config_path}")
            with open(config_path, "rt") as f:
                self.model = Model.from_config(json.load(f))
            self.model.load_weights(weight_path)
            self.digest = self.fetch_digest(weight_path)
            self.graph = tf.get_default_graph()
            logger.debug(f"model.py line132 loaded model digest = {self.digest}")
            return True
        else:
            logger.debug(f"model files does not exist at {config_path} and {weight_path}")
            return False

    #保存模型权重文件 
    def save(self, config_path, weight_path):
        logger.debug(f"save model to {config_path}")
        with open(config_path, "wt") as f:
            json.dump(self.model.get_config(), f)
            self.model.save_weights(weight_path)
        self.digest = self.fetch_digest(weight_path)
        logger.debug(f"saved model digest {self.digest}")

    #获取管道
    def get_pipes(self, num=1, api=None, need_reload=True):
        if self.api is None:
            self.api = CChessModelAPI(self.config, self)
            self.api.start(need_reload)
        return self.api.get_pipe(need_reload)

    #关闭管道
    def close_pipes(self):
        if self.api is not None:
            self.api.close()
            self.api = None

