from django.db import models


# 定义语料模型
class corpus(models.Model):
    # 语料ID
    corpus_ID = models.AutoField(primary_key=True)
    # 序号
    corpus_Num = models.IntegerField()
    # 语料情感极性
    corpus_emo = models.TextField()
    # 语料内容
    corpus_content = models.TextField()

    def __str__(self):
        return self.corpus_Num




