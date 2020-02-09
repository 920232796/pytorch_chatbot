# pytorch_chatbot
pytorch实现聊天机器人，seq2seq模型

使用数据： 链接:https://pan.baidu.com/s/1b5VnXY7_a25TjM_4cZx4QA  密码:j4mf 

encoder使用了双向LSTM，decoder使用单项LSTM，还使用了attention操作。相关教程文章请到：http://www.blog.zhxing.online/#/readBlog?blogId=302

数据下载下来直接放在data文件夹下面就ok。
1. 运行vocab.py 可以生成字典，不过本项目已经提前生成了，也就是vocab.json
2. 运行train.py 可以开始训练数据，我是训练了2个半小时，loss从83降到了33左右。
