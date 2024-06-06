# adafaceCosine
[![Python and MATLAB](https://img.shields.io/badge/Platforms-Python%20and%20MATLAB-blue.svg)](https://www.python.org/) [![MATLAB](https://img.shields.io/badge/MATLAB-EECS6.01-green.svg)](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-01sc-introduction-to-electrical-engineering-and-computer-science-i-spring-2011/)
In the field of FSR, pre-trained adaface is used to extract identity features and calculate cosine similarity, which is used to compare the help of FSR for downstream tasks

## Example

![Cosine](https://github.com/neverwinHao/adafaceCosine/blob/main/img/Cosine.png)

# Usage
配置好adaface的环境
```ba'sh
python inference.py
```
得到csv文件后，用plot.py画图
```ba'sh
python plot.py
```
