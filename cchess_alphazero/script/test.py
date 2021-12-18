conda create -n xq36 python=3.6.3

conda install -n env_name package_name

conda activate xq36

conda deactivate env_name

conda remove -n env_name --all

conda remove --name env_name  package_name 

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn

#恢复默认镜像渠道
conda config --remove-key channels


pip install -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple tensorflow-gpu==1.3.0

pip install -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple keras==2.0.8

pip install -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple Cython


pip install -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple numba

from sklearn.linear_model import LogisticRegression as lr

LogisticRegression( solver='lbfgs', penalty='l2', class_weight=None, tol=0.0001, random_state=None, C=1.0, fit_intercept=True, intercept_scaling=1, dual=False,  max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)


#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

netstat -lnp|grep 8892
ps xxxx 
kill -9 xxxx

conda activate tf2

python D:/projects/ChineseChess-AlphaZero-distributed/cchess_alphazero/run.py self --gpu '0'

python D:/projects/ChineseChess-AlphaZero-distributed/cchess_alphazero/run.py play --gpu '0'

python D:/projects/ChineseChess-AlphaZero-distributed/cchess_alphazero/run.py eval --gpu '0'

# python D:/projects/ChineseChess-AlphaZero-distributed/cchess_alphazero/run.py --type distribute --distributed self --gpu '0'

python D:/projects/ChineseChess-AlphaZero-distributed/cchess_alphazero/run.py opt --gpu '0'

1. 服务器部署， 如何解决， 全球范围用户访问网络延时， 或者网络不通（包括用户主动关闭本地网络）， 是否会影响用户体验？
2. 调用服务接口是否有次数限制， 能达到什么样的响应速度？
3. 项目实施周期大概需要多久? 如何衡量模型使用效果？是否有类似案例， 他们的效果怎么样？
4. 模型数据多久更新一次， 需要我们提供什么协助？需要哪些数据进行模型训练？


filtered_df = read_df.select(["_device_id","_ori_chnl", "_app_version", "_ip", "revenue", "ads_platform", "_day", "platid"])
agg_df = filtered_df.groupBy("_device_id").agg(
        F.sum(F.col("revenue")).alias("revenue_sum")
        ,F.count(F.col("revenue")).alias("revenue_count")
        ,F.max(F.col("_day")).alias("day_max")
        )


1. 炮二平五	　	炮8平5	　	2. 炮五进四	　	士4进5
如果把这4步棋涉及的5个局面都告诉引擎，那么指令依次是：
　
1: position fen rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1
2: position fen rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1 moves h2e2
3: position fen rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1 moves h2e2 h7e7
4: position fen rnbakabnr/9/1c2c4/p1p1C1p1p/9/9/P1P1P1P1P/1C7/9/RNBAKABNR b - - 0 2
5: position fen rnbakabnr/9/1c2c4/p1p1C1p1p/9/9/P1P1P1P1P/1C7/9/RNBAKABNR b - - 0 2 moves d9e8


"rkemsmekr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RKEMSMEKR"

#设置notebook显示100列
pd.set_option('display.max_columns', 100)
#显示400行
pd.set_option('display.max_rows', 400)
#设置默认单元格的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

import sklearn.metrics

y_pred = np.array(pred) 
y_real = np.array(data)
def score(y_pred,y_real):  #传入预测值和真实值

    def mape(y_real,y_pred): #结果要加%
        return 100.0*np.mean(np.abs((y_pred-y_real)/y_real))
    def smape(y_real,y_pred):  #结果要加%
        return 100.0*np.mean(2*np.abs(y_pred-y_real)/(np.abs(y_pred)+np.abs(y_real)))
    
    mse = metrics.mean_squared_error(y_real,y_pred) #均方误差
    rmse = np.sqrt(mse)  #均方根误差
    mae = metrics.mean_absolute_error(y_real,y_pred) #平均绝对误差
    mape = mape(y_real,y_pred)  #平均绝对百分比误差
    smape = smape(y_real,y_pred) #对称平均绝对百分比误差
    
    r2 = r2_score(y_real, y_pred) #R方

    print(f"均方误差为：{mse}\n均方根误差为：{rmse}\n平均绝对误差为：{mae}\n平均绝对百分比误差为：{mape}\n对称平均绝对百分比误差为：{smape}\nR方为：{r2}")


def calculate_r(rlist):
    result = predictAll(rlist)
    r1 = result.get('a365')[:len(rlist)]
    y = np.array(rlist)
    f = np.array(r1)
    #R方计算
    avg_r = y.mean()
    sst=0
    sse=0
    ssr=0
    for i in range(len(rlist)):
        sst += (y[i]-avg_r)**2
        sse += (y[i]-f[i])**2
        ssr += (f[i]-avg_r)**2
    print(f"R方：{ssr/sst}")
    
    r_adjusted = 1-(1-ssr/sst)*(len(rlist)-1)/(len(rlist)-1-1)
    print(f"调整后的R方：{r_adjusted}")


    pip install -r requirements.txt


    pip install -i http://pypi.douban.com/simple/ --trusted-host=pypi.douban.com/simple coloredlogs

    conda create -n pytorch39 python=3.9
    
    conda install -n env_name package_name
    
    conda activate env_name
    
    conda deactivate env_name
    
    conda remove -n env_name --all
    
    conda remove --name env_name  package_name 
    
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn
    
    #恢复默认镜像渠道
    conda config --remove-key channels


    #安装miniconda
    bash Miniconda3-py39_4.9.2-Linux-x86_64.sh
    
    source ~/.bashrc
    
    conda config --set auto_activate_base false

    conda install -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge coloredlogs