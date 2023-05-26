

蒙特卡洛模式的dqn

(区别于时序差分模式)

训练数据中,不需要 next_state
不需要 target 网络,所以,一共只需要一个网络.



cd cart
python run_this.py

td_on = 0 
为蒙特卡洛模式,不需要 next_state
不需要target网络.一共只需要一个网络.



td_on = 1
为时序差分模式.


两种模式均可训练出成果.本项目并没有比较分析两者的区别.也许收敛速度不同.

