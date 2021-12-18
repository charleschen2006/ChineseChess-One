python#后台执行flask程序， 日志以追加形式添加
nohup /opt/miniconda39/envs/pymc38_env/bin/python -u /home/dbt/project_dev/retentions.py >> /home/dbt/project_dev/retentions.log 2>&1 &
#查看存在的后台任务
jobs
#关闭后台任务
kill id号


#后台执行flask程序， 日志以追加形式添加
nohup python -u /home/dbt/project_dev/retentions.py >> /home/dbt/project_dev/predict.log 2>&1 &
#查看存在的后台任务
jobs
#关闭后台任务
kill id号

netstat -lnp|grep 5000
ps xxxx 
kill -9 xxxx

通过端口号查询进程号pId
ps -aux | grep 端口号
ps -ef  | grep 端口号
lsof -i:端口号
lsof -i | grep 端口号

通过服务器名称查看进程号pid
ps -aux/ef | grep 服务名称

根据进程查看此进程所占用的端口等信息
netstat -nap | grep pid
netstat -ntlp | grep pid

lsof -i | grep pid

此命令可以查端口和进程号 通过lsof -i:只能查端端口号

https://www.xqbase.com/protocol/cchess_ucci.htm
作者博客讲解
https://www.52coding.com.cn/2018/11/07/CCZero/

python D:/projects/ChineseChess-AlphaZero-distributed/cchess_alphazero/run.py self

python D:/projects/ChineseChess-AlphaZero-distributed/cchess_alphazero/run.py opt

netstat -lnp|grep 8088

state_ary length:342, policy_ary length:342, value_ary length:342


state_ary length:322, policy_ary length:322, value_ary length:322

state_ary[0] : [14 x 10 x 9]
policy_ary[0]: [0. 0. 0. ... 0. 0. 0.]
value_ary[0] : 1.0

var_name: cython.int

${var_name}: cython.double


import cython


@cython.cfunc
@cython.exceptval(check=True)
def ${func_name}():

    return 
