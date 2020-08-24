# -*-coding:utf-8 -*-
from django.conf.urls import url,include

# 引入视图文件
import text_emo.views

# 进行应用层次路由配置
urlpatterns = [
    url(r'^hello_world', text_emo.views.get_url),
    url(r'^index', text_emo.views.index),
    url(r'^search', text_emo.views.geturl),
    url(r'^selectNB', text_emo.views.NByes),
    url(r'^selectSVM', text_emo.views.SVM),
    url(r'^selectCNN', text_emo.views.CNN),
    url(r'^selectVOCAB', text_emo.views.VOCAB),
    url(r'^result', text_emo.views.ChartView.as_view()),
    url(r'^corpus', text_emo.views.get_corpus_page),  # 将语料分页
    url(r'^textclassify', text_emo.views.textclassify),
]