from collections import Counter  # 计算数组中元素的次数
from math import log  # log函数（应该是用于平滑）-避免算数下溢和提高计算速度
import jieba  # 分词

# (neg, pos) 的分类标记取为 (0, 1)，与各列表索引对应
train_files = ['D:\\python\\PycharmProjects\\Emo_classify\\text_emo\\static\\classify\\data\\1000\\neg_train.txt',
               'D:\\python\\PycharmProjects\\Emo_classify\\text_emo\\static\\classify\\data\\1000\\pos_train.txt']
# 训练数据集：100000正向 + 100000负向
# test_files = './data0/10000/pos_test.txt'
test_files = 'D:\\python\\PycharmProjects\\Emo_classify\\text_emo\\static\\classify\\data\\wb_comment_2.txt'
# 测试数据集


# 读取文件的函数
def read_lines(file):
    with open(file, encoding='utf-8') as f:  # 打开文件，编码为utf-8，作为变量f
        lines = f.readlines()  # 依次读取f的每一行
    return [line.strip() for line in lines]  # 返回每一行移除字符串头尾空格或换行符后的内容

nums = [len(read_lines(train_files[c])) for c in (0, 1)]  # 获取训练数据集的评论数量,c in (0,1)的意思是第一个数据集和第二个数据集
# len()方法是返回列表元素的个数，所以nums数组应该包含两个数：正面的评论数和负面的评论数
# print("the nums is : ", nums)
prior = [nums[c] / sum(nums) for c in (0, 1)]  # 训练数据先验概率 neg, pos，即P(c)
# print("the prior is : ", prior)  # 就目前训练数据而言，先验概率分别为0.5，0.5



# 读取训练数据，得到不同类别下的计数及词表
def get_count_and_vocab(files=train_files):
    count = [Counter(), Counter()]  # 计数：训练数据中的neg, pos分别为多少
    vocab = set()  # 词表
    # set()函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
    for c in (0, 1):
        for line in read_lines(files[c]):  # 依次读取训练集中的负面评论
            for word in jieba.cut(line):  # 利用结巴分词对每一条评论分词
                count[c][word] += 1  # 统计每一个分词出现的次数，此时的count相当于以一个字典
                vocab.add(word)  # 把分词加入到词表中
    # print("the count is : ", count)
    # print("the vocab is : ", vocab)
    return count, vocab



# 将计数转换为条件概率，采用 Laplace add1 平滑
def to_log_prob(count, vocab):  # 定义函数，输入的参数为count和vocab
    log_conditional = [Counter(), Counter()]  # 数组log_conditional包含两个元素：neg, pos
    vsize = len(vocab)  # 计算词表的元素个数:1315个分词
    # print("the vsize is : ", vsize)
    for c in (0, 1):
        total = sum(count[c].values())  # 统计neg和pos中各个词出现的总次数
        print("the total is : ", total)
        for word in vocab:  # 这里必须是 vocab 而不是 count[c].keys()，原因：.keys()返回的是字典中的键值，可能会包含标点符号等？
            log_conditional[c][word] = log(count[c][word] + 1) - log(total + vsize)  # 现在分别计算所有正(负)面评论中所有词汇的条件概率？？？
            # (count[c][word] + 1)：laplace add+1平滑
            # (total + vsize):
            # print("the log_conditional[c][word] is : ", word, log_conditional[c][word])
            # eg: 体检 -7.471363088187097
        # print("the log_conditional is : ", log_conditional)
    return log_conditional

count, vocab = get_count_and_vocab()  # 获取分词出现的次数，以及词表
log_conditional = to_log_prob(count, vocab)  # 获取条件概率



# 计算文本 docu 与分类 c 的联合概率(取对数)
def cal_joint_prob(docu, c):
    log_joint_prob = log(prior[c])  # 联合概率的计算方式：取对数
    words = jieba.cut(docu)  # 词表等于文本结巴分词后的内容
    for word in words:  # 遍历词表中的每一个分词
        if word in vocab:  # (参考 slp ch6.2, 仅考虑（训练集）词表内的词)如果该词在训练集词表中出现
            log_joint_prob += log_conditional[c][word]  # 则联合概率的计算方式为上述的条件概率(p(d|c)=p(f1|c)*p(f2|c)*p(f3|c)...p(fn|c),取对数后变为加)
    return log_joint_prob


# 对文本 docu 进行分类
def classify(docu):
    prob = [cal_joint_prob(docu, c) for c in (0, 1)]
    '''
    if prob[1] - prob[0] >= 0.8:  # 如果正向概率-负向概率大于0.8
        return 1  # 则为正向
    elif prob[0] - prob[1] >= 0.8:  # 如果负向概率-正向概率大于0.8
        return 0  # 则为负向
    else:
        # print(prob[1], prob[0])
        return 2  # 否则为中性
    '''
    if prob[1] > prob[0]:  # 如果正向概率大于负向概率
        return 1  # 则为正向
    elif prob[0] > prob[1] :  # 如果负向概率大于正向概率
        return 0  # 则为负向
    else:
        return 2  # 否则为中性

'''
results = []  # 分类结果 neg, pos,中性
for line in read_lines(test_files):  # 利用测试集进行分类
    results.append(classify(line))  # 将分类结果依次放入结果数组中
    # if classify(line) == 2:  # 输出分类为中性的评论
    # print(line)

#print(results)
'''