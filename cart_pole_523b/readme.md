﻿11:01 2023/5/23

不用差分时序，没有成品，只好自己改一个出来。

建议去看看policy gradient和model predictive control，都不用时序差分发布于 2021-08-23 22:58​赞同​​收起评论​分享​收藏​喜欢写下你的评论...1 条评论默认最新刘德柱提问者可能我问题阐述的有点问题，我这里指的是完全不借助时序差分这种机制本身的算法，即使是vanilla pg在强化信号的估计上依然使用了mc方法。比如两个月前刚发在arxiv上的decision transformer。就是完全不借助TD框架学习决策的模型。所以主要是想问问看有没有一些过去的论文，对强化学习的目标，是否有另外的解决方法。2021-09-11

作者：HammerWang
链接：https://www.zhihu.com/question/480946038/answer/2079565215
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



方向 就是，删。
把dqn的目标网络删掉。



11:09 2023/5/23

我一开始就不喜欢计算机。
现在呢，对cs稍有兴趣，但觉得软件工程只是乏味的工作。
我还是更对数学感兴趣。

现在想走人工智能，而我要更倾向学数学，算法与理论，少去掺和软件工程的问题（比如用什么库，封装哪个功能）。


我也关心这个 问题，我的需求是竞技对战。
1 求出所有可行解
2 结局时才结算奖励
我像个憨憨一样在改dqn，希望我好运


我现在有个疑惑，我是为啥要用mc而不用td的？？？
1   因为游戏每个step给不了奖励。结局时才结算。


成功

现在的任务 是，
任务2 
加入随机决策的逻辑。随机在最优的几个 解中取一个。

任务3 
用对战的形式来训练ai玩单机游戏 ，就像考试一样。


任务2
有太多 的思路 了，我要搞一种最简单的，不在这里浪费时间敲键盘。



mc td
cuda cpu
都写好了.

mc模式 import dqn
td模式 import dqn_td
