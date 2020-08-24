import os
import django
import codecs

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Emo_classify.settings')
django.setup()

from text_emo.models import corpus

#data_path = '../data/article'


def main():
    content_list = []
    i=1
    '''
    files = os.listdir(data_path)
    for name in files:
        f = os.path.join(data_path, name)
        with open(f, 'r', encoding='utf-8') as f:
            content = f.read()
            item = (name[:-4], content[:100], content)
            content_list.append(item)
    # Article.objects.all().delete()
    '''

    with codecs.open('../text_emo/static/classify/data/corpus282000.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            try:
                line = line.rstrip().split('\t')
                assert len(line) == 2
                if line[0]=="积极":
                    item = (i, "pos", line[1])
                elif line[0]=="消极":
                    item = (i, "neg", line[1])
                content_list.append(item)
                i=i+1
            except:
                pass
    #CorpusText.objects.all().delete()
    for item in content_list:
        #print('saving article: %s' % item[0])
        Corpus = corpus()
        Corpus.corpus_Num = item[0]
        Corpus.corpus_emo = item[1]
        Corpus.corpus_content = item[2]
        Corpus.save()


if __name__ == '__main__':
    main()
