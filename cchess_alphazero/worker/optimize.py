import os
import time
import gc
import subprocess
import shutil
import numpy as np

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from time import sleep
from random import shuffle
from threading import Thread

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.config import Config
from cchess_alphazero.lib.data_helper import get_game_data_filenames, read_game_data_from_file
from cchess_alphazero.lib.model_helper import load_best_model_weight, save_as_best_model
from cchess_alphazero.lib.model_helper import need_to_reload_best_model_weight, save_as_next_generation_model, save_as_best_model
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed, flip_policy, flip_move
from cchess_alphazero.lib.tf_util import set_session_config
from cchess_alphazero.lib.web_helper import http_request

# import tensorflow.compat.v1 as tf
import tensorflow as tf

# from keras.optimizers import SGD
from tensorflow.keras.optimizers import SGD
# from keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import TensorBoard
# from keras.utils import multi_gpu_model
# import keras.backend as K
import tensorflow.keras.backend as K

logger = getLogger(__name__)

def start(config: Config):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    return OptimizeWorker(config).start()

class OptimizeWorker:
    def __init__(self, config:Config):
        self.config = config
        self.model = None
        self.loaded_filenames = set()
        self.loaded_data = deque(maxlen=self.config.trainer.dataset_size)
        self.dataset = deque(), deque(), deque()
        self.executor = ProcessPoolExecutor(max_workers=config.trainer.cleaning_processes)
        self.filenames = []
        self.opt = None
        self.count = 0
        self.eva = False

    def start(self):
        self.model = self.load_model()
        self.training()

    def training(self):
        self.compile_model()
        total_steps = self.config.trainer.start_total_steps
        print(f"total_steps: {total_steps}")
        bef_files = []
        last_file = None

        while True:
            files = get_game_data_filenames(self.config.resource)
            offset = self.config.trainer.min_games_to_begin_learn
            if (len(files) < self.config.trainer.min_games_to_begin_learn \
              or ((last_file is not None and last_file in files) and files.index(last_file) + 1 + offset > len(files))):
                # if last_file is not None:
                #     logger.info('Waiting for enough data 300s, ' + str((len(files) - files.index(last_file)) * self.config.play_data.nb_game_in_file) \
                #             +' vs '+ str(self.config.trainer.min_games_to_begin_learn)+' games')
                # else:
                #     logger.info('Waiting for enough data 300s, ' + str(len(files) * self.config.play_data.nb_game_in_file) \
                #             +' vs '+ str(self.config.trainer.min_games_to_begin_learn)+' games')
                # time.sleep(300)
                if last_file is not None:
                    self.save_current_model(send=True)
                break
            else:
                if last_file is not None and last_file in files:
                    idx = files.index(last_file) + 1
                    if len(files) - idx > self.config.trainer.load_step:
                        files = files[idx:idx + self.config.trainer.load_step]
                    else:
                        files = files[idx:]
                elif len(files) > self.config.trainer.load_step:
                    files = files[0:self.config.trainer.load_step]
                last_file = files[-1]
                logger.info(f"Last file = {last_file}")
                logger.debug(f"files = {files[0:-1:2000]}")
                self.filenames = deque(files)
                logger.debug(f"Start training {len(self.filenames)} files")
                shuffle(self.filenames) #打乱文件名list顺序
                self.fill_queue()
                self.update_learning_rate(total_steps)
                if len(self.dataset[0]) > self.config.trainer.batch_size:
                    steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
                    total_steps += steps
                    self.save_current_model(send=False)
                    self.update_learning_rate(total_steps)
                    self.count += 1
                    #手动清理内存开始
                    a, b, c = self.dataset
                    a.clear()
                    b.clear()
                    c.clear()
                    del self.dataset, a, b, c
                    gc.collect()
                    #清理内存完毕

                    #创建新dataset对象
                    self.dataset = deque(), deque(), deque()
                    self.backup_play_data(files)

    def train_epoch(self, epochs):
        tc = self.config.trainer
        state_ary, policy_ary, value_ary = self.collect_all_loaded_data()
        print(f"state_ary length:{len(state_ary)}, policy_ary length:{len(policy_ary)}, value_ary length:{len(value_ary)}")
        print(f"state_ary[0] length:{len(state_ary[0])}")
        print(f"policy_ary[0] length:{len(policy_ary[0])}")
        print(f"value_ary[0]:{value_ary[0]}")
        tensorboard_cb = TensorBoard(log_dir="./logs", batch_size=tc.batch_size, histogram_freq=1)
        if self.config.opts.use_multiple_gpus: #多GPU计算
            self.mg_model.fit(state_ary, [policy_ary, value_ary],
                                 batch_size=tc.batch_size,
                                 epochs=epochs,
                                 shuffle=True,
                                 validation_split=0.02,
                                 callbacks=[tensorboard_cb])
        else:
            self.model.model.fit(state_ary, [policy_ary, value_ary],
                                 batch_size=tc.batch_size,
                                 epochs=epochs,
                                 shuffle=True,
                                 validation_split=0.02,
                                 callbacks=[tensorboard_cb])
        steps = (state_ary.shape[0] // tc.batch_size) * epochs
        return steps

    def compile_model(self):
        print(f"编译模型")
        # self.opt = SGD(lr=0.02, momentum=self.config.trainer.momentum)
        self.opt = SGD(learning_rate=0.02, momentum=self.config.trainer.momentum)
        losses = ['categorical_crossentropy', 'mean_squared_error']
        if self.config.opts.use_multiple_gpus:
            self.mg_model = multi_gpu_model(self.model.model, gpus=self.config.opts.gpu_num)
            self.mg_model.compile(optimizer=self.opt, loss=losses, loss_weights=self.config.trainer.loss_weights)
        else:
            print(f"执行编译")
            self.model.model.compile(optimizer=self.opt, loss=losses, loss_weights=self.config.trainer.loss_weights)

    def update_learning_rate(self, total_steps):
        # The deepmind paper says
        # ~400k: 1e-2
        # 400k~600k: 1e-3
        # 600k~: 1e-4
        logger.info(f"play data数据进行解析")
        lr = self.decide_learning_rate(total_steps)
        if lr:
            K.set_value(self.opt.lr, lr)
            logger.debug(f"total step={total_steps}, set learning rate to {lr}")

    def fill_queue(self):
        futures = deque()  #deque是一种能够在序列两端快速Pop和append操作的数据结构， 其时间复杂度为0(1)
        n = len(self.filenames) #统计总play data记录数
        #开启多进程
        with ProcessPoolExecutor(max_workers=self.config.trainer.cleaning_processes) as executor:
            for _ in range(self.config.trainer.cleaning_processes):
                if len(self.filenames) == 0:
                    break
                filename = self.filenames.pop()
                # logger.debug("loading data from %s" % (filename))
                futures.append(executor.submit(load_data_from_file, filename, self.config.opts.has_history))
            while futures and len(self.dataset[0]) < self.config.trainer.dataset_size: #fill tuples
                _tuple = futures.popleft().result()
                if _tuple is not None:
                    for x, y in zip(self.dataset, _tuple):
                        x.extend(y)
                m = len(self.filenames)
                if m > 0:
                    if (n - m) % 1000 == 0:
                        logger.info(f"Reading {n - m} files")
                    filename = self.filenames.pop()
                    # logger.debug("loading data from %s" % (filename))
                    futures.append(executor.submit(load_data_from_file, filename, self.config.opts.has_history))

    def collect_all_loaded_data(self):
        state_ary, policy_ary, value_ary = self.dataset

        state_ary1 = np.asarray(state_ary, dtype=np.float32)
        policy_ary1 = np.asarray(policy_ary, dtype=np.float32)
        value_ary1 = np.asarray(value_ary, dtype=np.float32)
        return state_ary1, policy_ary1, value_ary1

    def load_model(self):
        print(f"optimize load model")
        model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(model):
            print(f"optimize build model")
            model.build()
            save_as_best_model(model)
        return model

    def save_current_model(self, send=False):
        logger.info("Save as ng model")
        if not send:
            save_as_best_model(self.model)
        else:
            save_as_next_generation_model(self.model)

    def decide_learning_rate(self, total_steps):
        ret = None
        """
        self.lr_schedules = [
            (0, 0.03),
            (100000, 0.01),
            (200000, 0.003),
            (300000, 0.001),
            (400000, 0.0003),
            (500000, 0.0001),
        ]
        """
        #退火算法
        for step, lr in self.config.trainer.lr_schedules:
            if total_steps >= step:
                ret = lr
        return ret
    
    #尝试重新加载模型 
    def try_reload_model(self):
        logger.debug("check model")
        if need_to_reload_best_model_weight(self.model):
            with self.model.graph.as_default():
                load_best_model_weight(self.model)
            return True
        return False

    #备份游戏数据
    def backup_play_data(self, files):
        
        backup_folder = os.path.join(self.config.resource.data_dir, 'trained')
        logger.info("Backup play data to path: {backup_folder}")
        cnt = 0
        if not os.path.exists(backup_folder):
            os.makedirs(backup_folder)
        for i in range(len(files)):
            try:
                shutil.move(files[i], backup_folder)
            except Exception as e:
                # logger.error(f"Backup error : {e}")
                cnt = cnt + 1
        logger.info(f"backup {len(files)} files, {cnt} empty files")

#1. 加载文件中的游戏记录
def load_data_from_file(filename, use_history=False):
    try:
        logger.info(f"加载文件名称：{filename}")
        data = read_game_data_from_file(filename)
    except Exception as e:
        logger.error(f"Error when loading data {e}")
        os.remove(filename)
        return None
    if data is None:
        return None
    return expanding_data(data, use_history)

#2. 解析数据形成游戏记录
def expanding_data(data, use_history=False):
    logger.info(f"play data数据进行解析")
    state = data[0]
    real_data = []
    action = None
    policy = None
    value = None
    if use_history:
        history = [state]
    else:
        history = None
    for item in data[1:]:
        action = item[0]
        value = item[1]
        try:
            policy = build_policy(action, flip=False)
        except Exception as e:
            logger.error(f"Expand data error {e}, item = {item}, data = {data}, state = {state}")
            return None
        real_data.append([state, policy, value])
        state = senv.step(state, action)
        if use_history:
            history.append(action)
            history.append(state)
    # print(f"action: {action}")
    # print(f"state: {state}")    
    return convert_to_trainging_data(real_data, history)

#3. 根据游戏记录生成训练数据
def convert_to_trainging_data(data, history):
    logger.info(f"play data数据转化为训练数据")
    state_list = []  #状态数据
    policy_list = [] #政策数据
    value_list = []  #价值
    i = 0

    for state, policy, value in data:
        if history is None:
            state_planes = senv.state_to_planes(state)  #状态到数组
        else:
            state_planes = senv.state_history_to_planes(state, history[0:i * 2 + 1])
        sl_value = value

        state_list.append(state_planes)
        policy_list.append(policy)
        value_list.append(sl_value)
        i += 1
    
    # print(f"state_list: {state_list}")
    # print(f"policy_list len: {policy_list}")
    # print(f"value_list: {value_list}")
    return np.asarray(state_list, dtype=np.float32), \
           np.asarray(policy_list, dtype=np.float32), \
           np.asarray(value_list, dtype=np.float32)

# 策略头（policy head）输出：2086
# 2086 是行动空间的大小。行动空间就是说根据中国象棋的规则，任意棋子在任意位置的走法集合。
# 价值头（value head）输出：1
# 价值头输出一个标量衡量当前局势 v∈[−1,1]：当 v 接近1时，局势大好；接近0为均势；接近-1为败势。
def build_policy(action, flip):
    logger.info(f"根据action构建policy")
    labels_n = len(ActionLabelsRed)
    move_lookup = {move: i for move, i in zip(ActionLabelsRed, range(labels_n))}
    policy = np.zeros(labels_n)

    policy[move_lookup[action]] = 1

    if flip:
        policy = flip_policy(policy)
    return list(policy)






