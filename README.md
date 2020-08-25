# 简介
基于sequence-to-sequence的中文语音识别模型，解码采用beam search
# 使用方法
## 一句话语音识别
修改main.py中的path，可选的解码方式有：
1. greedy
   设置main.py中的lm_path=None, bw=None
2. beam search
   设置main.py中的lm_path=None, bw=4（或者8、16）
3. 加语言模型，second pass
   设置lm_path位置
