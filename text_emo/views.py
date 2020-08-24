# -*-coding:utf-8 -*-

from django.shortcuts import render
import os
from django.http import HttpResponse
import sys
from collections import Counter
from math import log
import jieba
import json
from random import randrange
from rest_framework.views import APIView
from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.charts import Pie
import joblib
import codecs
from text_emo.models import corpus
from django.core.paginator import Paginator
import chardet



sys.path.append(r'D:\python\PycharmProjects\Emo_classify\text_emo\static\classify\cnn')  # cnn的地址
import text_predict
sys.path.append(r'D:\python\PycharmProjects\Emo_classify\text_emo\static\classify\svm')  # SVM的地址
import svmpredict, Tools
sys.path.append(r'D:\python\PycharmProjects\Emo_classify\text_emo\static\classify')  # NByes的地址
# import NByes0
import NB3  # 导入朴素贝叶斯分类器
sys.path.append(r'D:\python\PycharmProjects\Emo_classify\text_emo\static\classify\vocab')  # vocab的地址
import classify



def get_url(request):
    return HttpResponse('hello word')


def index(request):  # 首页展示
    # search.html用于输入地址并进行爬虫
    return render(request, 'search.html', {'text': 'ran search successful'})

def textclassify(request):
    if request.method == "GET":
        request.encoding = 'utf-8'
        task_type = request.GET.get('task_type', '')
        comment = request.GET.get('comment', '')
        #comment = comment.decode()
        #comment = comment0.replace('\n', '').replace('\r', '')
        print(task_type,comment)
        result=''
        if task_type=="cnn":
            sentences = []
            sentences.append(comment)
            cat = text_predict.predict(sentences)
            result = cat[0]
            task_type="卷积神经网络"
            print(cat,result)
        elif task_type=="svm":
            with open(
                    'D:\\python\\PycharmProjects\\Emo_classify\\text_emo\\static\\classify\\svm\\test_corpus\\test\\test.txt',
                    "r+") as f:
                f.truncate()  # 清空文件
            with open(
                    "D:\\python\\PycharmProjects\\Emo_classify\\text_emo\\static\\classify\\svm\\test_corpus\\test\\test.txt",
                    "a", encoding="utf-8") as mon:
                mon.write(comment)
            corpus_path = "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/test_corpus/"  # 未分词分类语料库路径
            seg_path = "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/test_corpus_seg/"  # 分词后分类语料库路径
            svmpredict.corpus_segment(corpus_path, seg_path)

            # 对测试集进行Bunch化操作：
            wordbag_path = "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/test_word_bag/test_set.dat"  # Bunch存储路径
            seg_path = "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/test_corpus_seg/"  # 分词后分类语料库路径
            svmpredict.corpus2Bunch(wordbag_path, seg_path)

            stopword_path = "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/train_word_bag/hlt_stop_words.txt"
            bunch_path = "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/test_word_bag/test_set.dat"
            space_path = "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/test_word_bag/testspace.dat"
            train_tfidf_path = "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/train_word_bag/tfdifspace.dat"
            svmpredict.vector_space(stopword_path, bunch_path, space_path, train_tfidf_path)

            # 导入测试集
            testpath = "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/test_word_bag/testspace.dat"
            test_set = Tools.readbunchobj(testpath)
            clf = joblib.load(
                "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/model/my_linearSVC_model282000.m")
            # 预测分类结果
            predicted = clf.predict(test_set.tdm)
            if predicted[0]=='neg':
                result = "消极"
            elif predicted[0]=='pos':
                result = "积极"
            print(result)
            task_type = "支持向量机"
        elif task_type=="nb":
            a = NB3.classify(comment)
            if a==0:
                result = "消极"
            elif a==1:
                result = "积极"
            elif a==2:
                result = "中性"
            print(result)
            task_type = "朴素贝叶斯"
        elif task_type=="vocab":
            score = classify.setiment_score(comment)  # 调用vocab的分类函数
            if score > 0:
                result = "积极"
            elif score < 0:
                result = "消极"
            else:
                result = "中性"
            print(result)
            task_type = "基于情感词典"
        return render(request, 'textclassify.html', {
                                                'result':result,
                                                'comment': comment,
                                                "task_type":task_type
                                                })

def geturl(request):  # 获取爬取地址
    sys.path.append(r'D:\python\PycharmProjects\Emo_classify\text_emo\static\classify')
    import my_new_weibo  # 导入微博爬虫程序
    '''
    myurl = request.POST.get('search_url')
    username = "18387230662"  # 用户名
    password = "y$z&R1997%826"  # 密码
    cookie_path = "Cookie.txt"  # 保存cookie 的文件名称
    weibo = my_new_weibo.WeiboLogin(username, password, cookie_path)
    weibo.login()  # 登陆微博
    my_new_weibo.weibo_comment()
    my_new_weibo.get_comment(myurl)
    # print(myurl)
    '''
    return render(request, 'search.html', {'text': 'ran geturl successful'})

#CNN分类
def CNN(request):
    sentences=[]
    CNNdata = request.FILES.get('myfile')  # 获取到上传的文件
    # -*- coding=utf-8 -*-
    for line in CNNdata.readlines():
        line = line.decode('utf-8', 'ignore')
        try:
            sentences.append(line)
        except:
            pass
    print('predict test data.... ')
    cat = text_predict.predict(sentences)
    print("预测完毕!!!")
    num0 = 0  # neg
    num1 = 0  # pos
    num2 = 0  # 中性
    for i in range(len(cat)):
        if cat[i] == "消极":
            num0 = num0 + 1
        elif cat[i] == "积极":
            num1 = num1 + 1
    c = [num0, num1, num2]
    print(num0)
    print(num1)
    print(num2)
    print('get results successfully')
    return JsonResponse(json.loads(pie_base(c)))

#SVM分类器
def SVM (request):
    SVMdata = request.FILES.get('myfile')  # 获取到上传的文件
    results = []  # 用于存储结果
    # -*- coding=utf-8 -*-
    with open('D:\\python\\PycharmProjects\\Emo_classify\\text_emo\\static\\classify\\svm\\test_corpus\\test\\test.txt', "r+") as f:
        f.truncate()  # 清空文件
    with open("D:\\python\\PycharmProjects\\Emo_classify\\text_emo\\static\\classify\\svm\\test_corpus\\test\\test.txt",
              "a", encoding="utf-8") as mon:
        for line in SVMdata.readlines():
            line = line.decode('utf-8', 'ignore')
            mon.write(line.strip('\n'))
    corpus_path = "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/test_corpus/"  # 未分词分类语料库路径
    seg_path = "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/test_corpus_seg/"  # 分词后分类语料库路径
    svmpredict.corpus_segment(corpus_path, seg_path)
    # 对测试集进行Bunch化操作：
    wordbag_path = "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/test_word_bag/test_set.dat"  # Bunch存储路径
    seg_path = "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/test_corpus_seg/"  # 分词后分类语料库路径
    svmpredict.corpus2Bunch(wordbag_path, seg_path)
    stopword_path = "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/train_word_bag/hlt_stop_words.txt"
    bunch_path = "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/test_word_bag/test_set.dat"
    space_path = "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/test_word_bag/testspace.dat"
    train_tfidf_path = "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/train_word_bag/tfdifspace.dat"
    svmpredict.vector_space(stopword_path, bunch_path, space_path, train_tfidf_path)
    # 导入测试集
    testpath = "D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/test_word_bag/testspace.dat"
    test_set = Tools.readbunchobj(testpath)
    clf = joblib.load("D:/python/PycharmProjects/Emo_classify/text_emo/static/classify/svm/model/my_linearSVC_model282000.m")
    # 预测分类结果
    predicted = clf.predict(test_set.tdm)
    print("预测完毕!!!")
    num0 = 0  # neg
    num1 = 0  # pos
    num2 = 0  # 中性
    for i in range(len(predicted)):
        if predicted[i] == "neg":
            num0 = num0 + 1
        elif predicted[i] == "pos":
            num1 = num1 + 1
    c = [num0, num1, num2]
    print(num0)
    print(num1)
    print(num2)
    print('get results successfully')
    return JsonResponse(json.loads(pie_base(c)))


# 朴素贝叶斯分类器
def NByes(request):
    NBdata = request.FILES.get('myfile')  # 获取到上传的文件
    # print(type(NBdata))
    results = []  # 用于存储结果
    # -*- coding=utf-8 -*-
    for line in NBdata.readlines():
        line = line.decode('utf-8', 'ignore')
        #print(line)
        results.append(NB3.classify(line))  # 调用NByes的分类函数
    # print(NBdata.chunks())
    print(results)
    num0 = 0  # neg
    num1 = 0  # pos
    num2 = 0  # 中性
    for i in range(len(results)):
        if results[i] == 0:
            num0 = num0 + 1
        elif results[i] == 1:
            num1 = num1 + 1
        else:
            num2 = num2 + 1
    c = [num0, num1, num2]
    print(num0)
    print(num1)
    print(num2)
    print('get results successfully')
    # response = JsonResponse({"status": '服务器接收成功', 'data': [num0,num1]})
    return JsonResponse(json.loads(pie_base(c)))
    # return response
    # return render(request, 'search.html', {'text': 'ran nbyes0 successful'})

#情感词典分类
def VOCAB(request):

    VOCABdata = request.FILES.get('myfile')  # 获取到上传的文件
    # print(type(NBdata))
    results = []  # 用于存储结果
    # -*- coding=utf-8 -*-
    for line in VOCABdata.readlines():
        line = line.decode('utf-8', 'ignore')
        #print(line)
        score = classify.setiment_score(line)  # 调用vocab的分类函数
        if score > 0:
            score = "pos"
        elif score < 0:
            score = "neg"
        else:
            score = "neu"
        results.append(score)
    # print(NBdata.chunks())
    print(results)

    num0 = 0  # neg
    num1 = 0  # pos
    num2 = 0  # 中性
    for i in range(len(results)):
        if results[i] == "neg":
            num0 = num0 + 1
        elif results[i] == "pos":
            num1 = num1 + 1
        else:
            num2 = num2 + 1
    c = [num0, num1, num2]
    print(num0)
    print(num1)
    print(num2)
    print('get results successfully')
    # response = JsonResponse({"status": '服务器接收成功', 'data': [num0,num1]})
    return JsonResponse(json.loads(pie_base(c)))
    # return response
    # return render(request, 'search.html', {'text': 'ran nbyes0 successful'})


# 注: 目前由于json数据类型的问题，无法将pyecharts中的JSCode类型的数据转换成json数据格式返回到前端页面中使用
# 因此在使用前后端分离的情况下尽量避免使用JSCode进行画图
def response_as_json(data):
    json_str = json.dumps(data)
    response = HttpResponse(
        json_str,
        content_type="application/json",
    )
    response["Access-Control-Allow-Origin"] = "*"
    return response


def json_response(data, code=200):
    data = {
        "code": code,
        "msg": "success",
        "data": data,
    }
    return response_as_json(data)


def json_error(error_string="error", code=500, **kwargs):
    data = {
        "code": code,
        "msg": error_string,
        "data": {}
    }
    data.update(kwargs)
    return response_as_json(data)


JsonResponse = json_response
JsonError = json_error


def drawNB() -> Bar:
    c = (
        Bar()
        .add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
        .add_yaxis("商家A", [randrange(0, 100) for _ in range(6)])
        .add_yaxis("商家B", [randrange(0, 100) for _ in range(6)])
        .set_global_opts(title_opts=opts.TitleOpts(title="Bar-基本示例", subtitle="我是副标题"))
        .dump_options()
    )
    return c


def pie_base(b) -> Pie:  # 绘制饼图
    c = (
        Pie()
        .add("", [('负面', b[0]), ('正面', b[1]), ('中性', b[2])])
        .set_global_opts(title_opts=opts.TitleOpts(title=""))
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%"))
        .dump_options()
    )
    return c

# 饼图展示初始化
class ChartView(APIView):
    print('aaa')
    def get(self, request, *args, **kwargs):
        print('bbb')
        c = [50, 50, 50]
        return JsonResponse(json.loads(pie_base(c)))


class IndexView(APIView):
    def get(self, request, *args, **kwargs):
        return HttpResponse(content=open("./templates/search.html").read())

# 语料分页
def get_corpus_page(request):
    page = request.GET.get('page')  # 获取page参数
    if page:
        page = int(page)
    else:
        page = 1
    all_corpus = corpus.objects.all()[:1000]# 把前1000条语料取出来
    paginator = Paginator(all_corpus, 80)  # 80为每一页的数量
    page_num = paginator.num_pages
    page_corpus_list = paginator.page(page)  # 文章列表
    if page_corpus_list.has_next():
        next_page = page + 1
    else:
        next_page = page

    if page_corpus_list.has_previous():
        pre_page = page - 1
    else:
        pre_page = page

    return render(request, 'corpus.html',
              {
                  'corpus_list': page_corpus_list,
                  'page_num': range(1, page_num + 1),
                  'curr_page': page,
                  'next_page': next_page,
                  'pre_page': pre_page
              })


