

import pandas as pd
import re
import pymorphy2
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
    )
import datetime
segmenter = Segmenter()
morph = pymorphy2.MorphAnalyzer()

### Подключение стоп-слов
path_to_sw = 'C:\\Users\\Fedorova-VR\\chat_analysis\\nltk_data\\stopwords\\russian_sw.txt'
f = open(path_to_sw,'r',encoding = 'utf-8')
sw_data = f.read()
f.close()
stopwords = sw_data.strip().split('\n')

symbols_str = ',.?/\!@#$%^&*()-_=+":'

stopwords = stopwords + list(symbols_str)


### Стэммер Портера
class Porter:
    PERFECTIVEGROUND = re.compile(u"((ив|ивши|ившись|ыв|ывши|ывшись)|((?<=[ая])(в|вши|вшись)))$")
    REFLEXIVE = re.compile(u"(с[яь])$")
    ADJECTIVE = re.compile(u"(ее|ие|ые|ое|ими|ыми|ей|ий|ый|ой|ем|им|ым|ом|его|ого|ему|ому|их|ых|ую|юю|ая|яя|ою|ею)$")
    PARTICIPLE = re.compile(u"((ивш|ывш|ующ)|((?<=[ая])(ем|нн|вш|ющ|щ)))$")
    VERB = re.compile(u"((ила|ыла|ена|ейте|уйте|ите|или|ыли|ей|уй|ил|ыл|им|ым|ен|ило|ыло|ено|ят|ует|уют|ит|ыт|ены|ить|ыть|ишь|ую|ю)|((?<=[ая])(ла|на|ете|йте|ли|й|л|ем|н|ло|но|ет|ют|ны|ть|ешь|нно)))$")
    NOUN = re.compile(u"(а|ев|ов|ие|ье|е|иями|ями|ами|еи|ии|и|ией|ей|ой|ий|й|иям|ям|ием|ем|ам|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|ья|я)$")
    RVRE = re.compile(u"^(.*?[аеиоуыэюя])(.*)$")
    DERIVATIONAL = re.compile(u".*[^аеиоуыэюя]+[аеиоуыэюя].*ость?$")
    DER = re.compile(u"ость?$")
    SUPERLATIVE = re.compile(u"(ейше|ейш)$")
    I = re.compile(u"и$")
    P = re.compile(u"ь$")
    NN = re.compile(u"нн$")


    def stem(word):
        word = word.lower()
        word = word.replace(u'ё', u'е')
        m = re.match(Porter.RVRE,word)
        if m and m.groups():
            pre = m.group(1)
            rv = m.group(2)
            temp = Porter.PERFECTIVEGROUND.sub('',rv, 1)
            if temp == rv:
                rv = Porter.REFLEXIVE.sub('',rv, 1)
                temp = Porter.ADJECTIVE.sub('',rv, 1)
                if temp != rv:
                    rv = temp
                    rv = Porter.PARTICIPLE.sub('',rv, 1)
                else:
                    temp = Porter.VERB.sub('',rv, 1)
                    if temp == rv:
                        rv = Porter.NOUN.sub('',rv, 1)
                    else:
                        rv = temp
            else:
                rv = temp
            rv = Porter.I.sub('',rv, 1)

            if re.match(Porter.DERIVATIONAL,rv):
                rv = Porter.DER.sub('',rv, 1)

            temp = Porter.P.sub('',rv, 1)
            if temp == rv:
                rv = Porter.SUPERLATIVE.sub('',rv, 1)
                rv = Porter.NN.sub(u'н',rv, 1)
            else:
                rv = temp
            word = pre+rv

        return word
    
    stem=staticmethod(stem)
    
    

### Функция для замены аббревиатур и частых словосочетаний
def replace_abbr(text):
    text = text.lower()
    text = text.replace('пдкпдкп', 'договор купли продажи')
    text = text.replace('куплипродажи', 'купли продажи')
    text = text.replace('мат капитал', 'материнский капитал')
    text = text.replace('пдкп', 'договор купли продажи')
    text = text.replace('дкп', 'договор купли продажи')
    text = text.replace('дду', 'договор долевого участия')
    text = text.replace('приемапередачи', 'приема передачи')
    text = text.replace('добрый день', '')
    text = text.replace('добрый вечер', '')
    text = text.replace('доброе утро', '')
    text = text.replace('чат', '')
    text = text.replace('спасибо', '')
    text = text.replace('пожалуйста', '')
    text = text.replace('здравствуйте', '')
    text = text.replace('можете', '')
    text = text.replace('это', '')
    text = text.replace(' г ', '')
    return text

### Очистка текста от ссылок, чисел, стоп-слов и имен, а также токенизация
def clean_tokens(text):
    for sw in stopwords:
        text = text.replace(' ' + sw + ' ', ' ')
    text = re.sub(r'http[\S]+','',text)
    text = re.sub(r'\d+','',text)
    text = text.split()
    text = ' '.join([s for s in text if not ('Name' in morph.parse(s)[0].tag or 'Surn' in morph.parse(s)[0].tag or 'Patr' in morph.parse(s)[0].tag)])
    return text

def stem_token(tokens):
    for i in range(len(tokens)):
        tokens[i] = Porter.stem(tokens[i])
    return tokens

### Функция для поиска документа 
def search_docs(tokens):
    res_dict = {}
    res = []
    for t in tokens:
        if [t] in documents_one_grams:
            if res_dict.get(t):
                res_dict[t] += 1
            else:
                res_dict[t] = 1
    token_bigrams = ngrams(tokens,2)
    for gram in token_bigrams:
        if list(gram) in documents_bigrams:
            if res_dict.get(gram):
                res_dict[gram] += 1
            else:
                res_dict[gram] = 1
    token_3grams = ngrams(tokens,3)
    for gram in token_3grams:
        if list(gram) in documents_3grams:
            if res_dict.get(gram):
                res_dict[gram] += 1
            else:
                res_dict[gram] = 1
    token_4grams = ngrams(tokens,4)
    for gram in token_4grams:
        if list(gram) in documents_4grams:
            if res_dict.get(gram):
                res_dict[gram] += 1
            else:
                res_dict[gram] = 1
    
    for d in document_list_tokens:
        val = res_dict.get(tuple(d)) 
        val_one_gram = None
        if len(d) == 1:
            val_one_gram = res_dict.get(str(d[0]))
        if val is None and val_one_gram is None:
            res.append(0)
        else:
            res.append(1)
    return res


document_list = [
    'право собственности', 
    'договор дарения',
    'договор купли продажи', #'дкп'
    'свидетельство право наследства',
    'выписка егрн',
    'отчет оценка',
    'приема передачи',
    'технический паспорт',
    'поэтажный план',
    'экспликация',
    'брачный договор',
    'согласие супруги',
    'свидетельство рождении',
    'паспорт',
    'выписка домовой книги',
    'наличие зарегистрированных',
    'справка отсутствии зарегистрированных',
    'снилс',
    'инн',
    'реквизиты счета продавца',
    'договор основание приобретения',
    'справки жилищноэксплуатационного органа',
    'доверенность',
    'согласие опеки', 
    'органов опеки попечительства',
    'заявление залогодателя',
    'справку банка текущей задолженности',
    'наличие прописки',
    'выписку росреестра',
    'нотариально удостоверенныи отказ',
    'смета строительства',
    'пояснительная записка',
    'договороснование строительства',
    'домадоговор подряда',
    'разрешение строительства',
    'полис страхования',
    'кредитный договор',
    'справка использовании мат капитала',
    'справка неиспользовании мат капитала',
    'справка банка подтверждении кредита',
    'справка  рефинансировании',
    'справка пфр',
    'нотариальное свидетельство',
    'остатке задолженности рефинансируемому кредиту',
    'просроченной задолженности',
    'первичный кредитор',
    'свидетельство регистрации брака',
    'счет бик',
    'огрн',
    'справка зарегистрированных лицах',
    'договор приватизации',
    'отказ приватизации',
    'договор долевого участия', #,'дду'
    'закладная',
    'материнский капитал',
    'копия письма застройщика',
    'исковое заявление клиента к застройщику',
    'дополнительное соглашение договору подряда',
    'регистрация прав недвижимое имущество'
]

### Преобразование списка документов в список n-gram
from nltk import ngrams
document_list_tokens = []
for d in document_list:
    document_list_tokens.append([Porter.stem(el) for el in d.split()])

documents_one_grams = [el for el in document_list_tokens if len(el) == 1]
documents_bigrams = [el for el in document_list_tokens if len(el) == 2]
documents_3grams = [el for el in document_list_tokens if len(el) == 3]
documents_4grams = [el for el in document_list_tokens if len(el) == 4]


pm_data = pd.read_csv('for_pm.csv',sep=';')


pm_data['message'] = pm_data['message'].apply(replace_abbr)
pm_data['message'] = pm_data['message'].apply(clean_tokens)
pm_data['message'] = pm_data['message'].apply(stem_token)

pm_data['docs'] = pm_data['message'].apply(search_docs)

### Выделение сообщений, в которых нашлись документы
pm_data['found_flag'] = pm_data['docs'].apply(lambda x:set(x))

needed_chats = pm_data[pm_data['found_flag']!={0}]
needed_chats = needed_chats.sort_values(by=['chat_room_id','created_at'])

first_time = datetime.datetime.now()

def get_doc_token(vec):
    res = []
    for i in range(len(vec)):
        if vec[i] == 1:
            res.append(document_list[i])
    return res

### Получение текстового представления доумента из векторного
needed_chats['docs_token'] = needed_chats['docs'].apply(get_doc_token)

### Фильтрация коротких чатов
needed_chats_sizes = needed_chats.groupby('chat_room_id').size()
small_chats = needed_chats_sizes.loc[needed_chats_sizes < 2].index.tolist()
needed_chats = needed_chats.loc[~needed_chats['chat_room_id'].isin(small_chats)]
graph_data = needed_chats[['chat_room_id','docs_token']].values.tolist()

### Подсчет переходов по ребрам графа
graph_edges = {}
cnt = 0
first_edge = set()
second_edge = set()
for row in graph_data:
    if cnt == 0:
        cur_id = row[0]
        for i in range(len(row[1])):
            first_edge.add(row[1][i])
        cnt += 1
    else:
        new_id = row[0]
        if new_id != cur_id:
            first_edge = set()
            for i in range(len(row[1])):
                first_edge.add(row[1][i])
            cur_id = new_id
        else:
            for i in range(len(row[1])):
                second_edge.add(row[1][i])
            for first_el in first_edge:
                for second_el in second_edge:
                    if graph_edges.get(tuple([first_el,second_el])) is not None:
                        graph_edges[tuple([first_el,second_el])] += 1
                    else:
                        graph_edges[tuple([first_el,second_el])] = 1
            
            first_edge = second_edge
            second_edge = set()
            cur_id = new_id
            
            
### Фильрация редких ребер + сохранение порядка запроса документов       
common_edges = {key: value for key, value in graph_edges.items() if value > 10}        
common_edges_list = list(common_edges.keys())         
graph_data_new = []
cnt = 0
for row in graph_data:
    if cnt == 0:
        cur_id = row[0]
        for i in range(len(row[1])):
            first_edge.add(row[1][i])
        cnt += 1
        cur_time = first_time
    else:
        new_id = row[0]
        if new_id != cur_id:
            first_edge = set()
            for i in range(len(row[1])):
                first_edge.add(row[1][i])
            cur_id = new_id
            cur_time = first_time
        else:
            for i in range(len(row[1])):
                second_edge.add(row[1][i])
            sec_time = cur_time + datetime.timedelta(0,1)
            for first_el in first_edge:
                for second_el in second_edge:
                    if tuple([first_el,second_el]) in common_edges_list:
                        graph_data_new.append([new_id,first_el,cur_time])
                        graph_data_new.append([new_id,second_el,sec_time])
            
            first_edge = second_edge
            second_edge = set()
            cur_id = new_id
            cur_time = sec_time
            

cnt = 0
line = 'case:concept:name;concept:name;time:timestamp\n'
for row in graph_data_new:
    line += row[0] + ';' + row[1]  + ';' + row[2].strftime("%Y-%m-%d %H:%M:%S") + '\n'
    
             
res_file = open('C:\\Users\\Fedorova-VR\\chat_analysis\\graph_data.csv','w',encoding='utf-8')    
res_file.write(line)
res_file.close()
    



