# python部分

## sorted()

对所有可迭代的对象进行排序操作。

```python
sorted(iterable, cmp=None, key=None, reverse=False)
```

参数说明：

- iterable -- 可迭代对象。
- cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
- key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
- reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。

返回值

返回重新排序的列表。



## sort()

```python
list.sort( key=None, reverse=False)
```

- key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
- reverse -- 排序规则，**reverse = True** 降序， **reverse = False** 升序（默认）。



example:

```python
#!/usr/bin/python
 
# 获取列表的第二个元素
def takeSecond(elem):
    return elem[1]
 
# 列表
random = [(2, 2), (3, 4), (4, 1), (1, 3)]
 
# 指定第二个元素排序
random.sort(key=takeSecond)
 
# 输出类别
print ('排序列表：', random)
```



## array ->dict 

### enumerate()

将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。

eg:

![image-20230406164827891](C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230406164827891.png) 







## 生成器

**调用一个生成器函数，返回的是一个迭代器对象。**

在调用生成器运行的过程中，每次遇到 yield 时函数会暂停并保存当前所有的运行信息，返回 yield 的值, 并在下一次执行 next() 方法时从当前位置继续运行。



eg:

![image-20230406205745876](C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230406205745876.png)











## strip()

> 移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。

```python
str.strip([chars]);
```

## sum()



```python
sum(iterable[, start])
```

> 对序列进行求和计算。

### 参数

- iterable -- 可迭代对象，如：列表、元组、集合。
- start -- 指定相加的参数，如果没有设置这个值，默认为0。

### 返回值

返回计算结果。



sum() 也可以用于列表的展开，效果相当于各子列表相加

```python
lst = [[1, 2], [3, 4]]
print(sum(lst, [])) 
#[1, 2, 3, 4]
```





## copy()

> 构建完全独立的副本

````python
import copy 
copy.copy()
````



## **str.format()**

> 通过 **{}**格式化字符串

```python
>>>"{} {}".format("hello", "world")    # 不设置指定位置，按默认顺序
'hello world'
 
>>> "{0} {1}".format("hello", "world")  # 设置指定位置
'hello world'
 
>>> "{1} {0} {1}".format("hello", "world")  # 设置指定位置
'world hello world'


print("网站名：{name}, 地址 {url}".format(name="菜鸟教程", url="www.runoob.com"))
 
# 通过字典设置参数
site = {"name": "菜鸟教程", "url": "www.runoob.com"}
print("网站名：{name}, 地址 {url}".format(**site))
 
# 通过列表索引设置参数
my_list = ['菜鸟教程', 'www.runoob.com']
print("网站名：{0[0]}, 地址 {0[1]}".format(my_list))  # "0" 是必须的


```





----



# NumPy

+ **核心：ndarray**

N 维数组对象 ，它是一系列同类型数据的集合【存放同类型元素的多维数组】，以 0 下标为开始进行集合中元素的索引。

![image-20230406163358944](C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230406163358944.png)

+ 创建方式：

## 创建ndarray -> numpy.array

![image-20230406163600684](C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230406163600684.png)



eg:

```python
a = np.array([[1,2,3],[3,4,5],[4,5,6]])
```





自定义dtype:

![image-20230406165124007](C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230406165124007.png)

eg:







## 创建数组

+ numpy.zeros

创建指定大小的数组，数组元素以 0 来填充：

![image-20230406164138421](C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230406164138421.png)



eg:

```python
import numpy as np
y = np.zeros((3,3), dtype = int) 
print(y)
[[0,0,0]
 [0,0,0]
 [0,0,0]]
```



## 转置

```python
nparray对象.T
```







## 索引

x=np.array([[1,  2],  [3,  4],  [5,  6]]) 

get:

x[[0,1,2],  [0,1,0]]  ->[1  4  5]

x[1,2] =>4

set:



## 运用pandas.DataFrame()使数据可视化



# Pandas

## Pandas Series 

类似表格中的一个列（column），类似于一维数组，可以保存任何数据类型。

<img src="C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230406184802547.png" alt="image-20230406184802547" style="zoom:80%;" /> 

eg:   array -> Series

<img src="C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230406184825993.png" alt="image-20230406184825993" style="zoom:80%;" />



eg:     dict -> Series

![image-20230406185204004](C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230406185204004.png)





## pandas.DataFrame()

![image-20230406170633519](C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230406170633519.png)

```python
pandas.DataFrame(data, index, columns, dtype, copy)
```

![image-20230406170751968](C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230406170751968.png) 

```python
    DataFrame.from_records(data, index=None, exclude=None, columns=None, coerce_float=False, nrows=None)[source]
```

> 将结构化或记录的 ndarray 转换为 DataFrame。 从结构化的 ndarray、元组或字典序列或 DataFrame 创建 DataFrame 对象。
>
> 

Parameters

- **data： **structured ndarray, sequence of tuples or dicts, or DataFrame

  Structured input data.

- **index：**str, list of fields, array-like

  Field of array to use as the index, alternately a specific set of input labels to use.

- **exclude：**sequence, default None

  Columns or fields to exclude.

- **columns：**sequence, default None

  Column names to use. If the passed data do not have names associated with them, this argument provides names for the columns. Otherwise this argument indicates the order of the columns in the result (any names not found in the data will become all-NA columns).

- **coerce_float:**bool, default False

  Attempt to convert values of non-string, non-numeric objects (like decimal.Decimal) to floating point, useful for SQL result sets.

- **nrows:**int, default None

  Number of rows to read if data is an iterator.

Returns

- DataFrame





## 数据清洗

+ ### fillna()

> 替换一些空字段



----







# nlpia

> 多维数据的可视化









----

# 线性代数

+ 拉伸

![image-20230410121917095](C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230410121917095.png)





 应用：NLP的SVD模型

S的值只要从高到低逐渐减小，就能提高sort排序后前几位的重要性



+ 旋转

<img src="C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230410122006522.png" alt="image-20230410122006522" style="zoom:80%;" />



> R ->正交矩阵 ->使原矩阵旋转















































----



# 词袋向量

Series作为词袋向量，dataFrame负责封装该向量



+ **找出词袋向量间的共享词**

```python
[(k,v) for (k,v) in (df.sent0 &df.sent3).items() if v]
```

> & - 词袋向量之间进行&运算



+ **求重合度 - dot()**

```python
df.sent0.dot(df.sent1)
```









# 分词 - NLTK

重点：语句拆分，大小写转换，n-gram,标点符号的去除，停用词的去除,消除词的复数形式，所有格的词尾[a mile’s distance 一英里的距离中的`'s`]，不同的动词形式的归一化





:one:使用**正则表达式**来split词

split的依据：标点符号和空格 ->  [-\s.,;!?]

2️⃣**列表解析式**

[x for x in tokens if x and xnot in '- \t\n.,;!?']

> 利用for和if来组装，但in后的不是正则表达式，所以里面有个空格来表示空格



以上都不如使用**nltk(Natural Language Tookit)**



+ NLTK.RegexpTokenizer



+ NLTK.TreebankWordTokenizer【推荐】
+ 包含一些英文缩词 don't -> do 和n't

eg: 使用

```python
from nltk.tokenize import TreebankWordTokenizer
tokenizer=TreebankWordTokenizer();
tokenizer.tokenize(..);
```



+ casual_tokenize
+ 处理社交网络上的非规范的包含表情符号的短文本，剥离文本的用户名，减少词条内的重复字符数

eg:使用

```python
from nltk.tokenize.casual import casual_tokenize
casual_tokenize(message);
```





3️⃣n-gram的生成

+ nltk的ngrams()





eg:使用

```python
ngrams(alist,n);
```

参数说明：alist ->一个列表

​					n ->生成n-gram

返回值：一个生成器





n-gram生成器 -> list(tuple)

```python
list(ngrams(token,3));
```

list(tuple) -> list(string)

```python
[" ".join(X) for x in two_grams]
```







+ 停用词表

```python
import nltk
nltk.download("stopwords")
stop_words=nltk.corpus.stopwords.words("english")
```





4️⃣**大小写转换**

采用**列表解析式**

```python
[x.lower() for x in tokens]
```

:five:词形归并，使用词性来提高准确率

> 在词干还原之前使用词形归并有助于提高准确率



```python
import nltk
nltk.download('wordnet');
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer();
a=lemmatizer.lemmatize("goods",pos="a") -[a表示形容词]-> goods
lemmatizer.lemmatize("goods",pos="n") -[n表示名词]->good
```













:six:**词干还原**

+ nltk.stem.porter

```python
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer();
[stemmer.stem(w) for w in "....".split()]
```

> Porter会保留词尾的撇号`'`，将所有格形式和非所有格形式的词区分开5









7️⃣**分析词的情感**

+ 基于规则【关键词】

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sa=SentimentIntensityAnalyzer()
# sa.lexicon   ->评分表
# 评分
score=sa.polarity_scores(text="Python is very readable and it's great for NLP")
# {'neg': 0.0, 'neu': 0.661, 'pos': 0.339, 'compound': 0.6249}-》 neg[负向],neu[中立],pos[正向],compound[总得分]
```



+ 基于机器学习的模型，利用一系列已经标注好情感分的文档【数据集】来让机器学习



:eight:**分词后的计数**

+ Counter
+ 输入是一个列表,返回一个字典{对象:统计后的数量}

```python
from collections import Counter
Counter(casual_tokenize(text))
```







---



# TF-IDF

:one:计算TF[词项频率]，来判断词项在词袋中的重要性

+ Counter

统计词的频率

+ Counter(...).most_common(number)

参数说明：number->展示频率最高的number个词条





+ 以TF-IDF作为多维中一维的数值

![image-20230408163912895](C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230408163912895.png)





## TF-IDF向量生成

```python
import  pandas as pd
from nlpia.data.loaders import get_data

pd.options.display.width=120
# nlpia库中提供了数据集
sms=get_data('sms-spam')
# sms-spam -》csv文件
# ,spam,text
# 0,0,"Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."
# 1,0,Ok lar... Joking wif u oni...
# 2,1,Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
# 3,0,U dun say so early hor... U c already then say...
# print(sms)
index = ["sms{}{}".format(i,'!'*j) for(i,j) in zip(range(len(sms)),sms.spam)]
sms=pd.DataFrame(sms.values,columns=sms.columns,index=index)
sms["spam"]=sms.spam.astype(int)
print(sms.head(6))

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
# 以casual_tokenize作为分词方式,将原始文档集合转换为 TF-IDF 特征矩阵。
tfidf_model=TfidfVectorizer(tokenizer=casual_tokenize)
# sms.text是dataFrame中为text的那一列
tfidf_docs=tfidf_model.fit_transform(raw_documents=sms.text).toarray();




```





# 相似度

+ 基于词频的相识度

余弦相似度，可参考《自然语言处理实战利用python理解，分析和生成文本》P72

利用两个向量的夹角的余弦值来判断相似度





# 主题向量

> 词汇表中每个词都有自己的词-主题向量，文档中所有词的词-主题向量相加即可得到文档的主题向量

挑战：

![image-20230409110822973](C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230409110822973.png)





# LSA -隐性语义分析

> 维度变为了各种"主题"的词的加权组合

类似于TF-IDF中的IDF部分 ->  丢弃文档组件方差较小的维度[主题]，因为方差小就难以帮助我们去区分文档

## SVD

> 将相关度高（经常出现在相同文档）的词项组合在一起，同时这一组合在一组文档中出现的差异很大，我们认为这些词的线性组合就是主题。这些主题会将词袋向量（或TF-IDF向量）转换为主题向量

<img src="C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230410123941812.png" alt="image-20230410123941812" style="zoom: 25%;" />  



-> S最后一行数据无效，可去掉

<img src="C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230410124038015.png" alt="image-20230410124038015" style="zoom:25%;" /> 

->逐渐将S中不重要的维度给去掉

<img src="C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230410124117625.png" alt="image-20230410124117625" style="zoom:25%;" /> 





分解的意义，核心 -> 模型1已经能大概代表原来的矩阵了

<img src="C:\Users\DING\AppData\Roaming\Typora\typora-user-images\image-20230410124854795.png" alt="image-20230410124854795" style="zoom:25%;" /> 





### 截断的SVD模型

> 用于稀疏矩阵【词袋和TF-IDF几乎总是稀疏的】

### sckit-learn PCA模型

> 使用前需要将稀疏矩阵给填充了









# LDA - 分类器

> 作用：区分两类信息
>
> 在《自然语言处理实战利用python理解，分析和生成文本》以区分垃圾消息和非垃圾消息做示例







# 模型训练

如果训练集有25%的正向样本和75%的负向样本，同样期望测试集和验证集也有25%的正向样本和75%的负向样本

交叉拟合训练









# 概念

降维  ->  减少词汇表的规模

召回率 -> 度量搜索引擎返回所有相关文档的程度的一个指标





# question

question: ImportError: cannot import name 'Mapping' from 'collections'

method:

```python
#将
from collections import Mapping 
from collections import MutableMapping 
#这些导入全部改为以下这些
from collections.abc import Mapping
from collections.abc import MutableMapping
#即用collections.abc代替collections调用方法Mapping、MutableMapping
```

具体参考https://blog.csdn.net/LSH1628340121/article/details/124140926?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168101374516800182168289%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168101374516800182168289&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-124140926-null-null.142
