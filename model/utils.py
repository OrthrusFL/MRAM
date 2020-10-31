import gc
import re
import json
import requests
import os
import random
import warnings
import string
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from nltk.tokenize import RegexpTokenizer

warnings.filterwarnings('ignore')



stop_words = stopwords.words('english')
java_keywords = ["abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class",
                 "const", "continue", "default", "do", "double", "else", "enum", "extends", "false",
                 "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof",
                 "int", "interface", "long", "native", "new", "null", "package",
                 "private", "protected", "public", "return", "short", "static", "strictfp",
                 "super", "switch", "synchronized", "this", "throw", "throws",
                 "transient", "try", "true", "void", "volatile", "while"]


def format_time(old_time):
    new_time = datetime.strptime(old_time, "%Y-%m-%d %H:%M:%S")
    return new_time


def format_api_seq(api_seq):
    if api_seq == '':
        return api_seq

    api_list = api_seq.split('|')
    api_seq_formatted = ''
    for api in api_list:
        api = api.split('(')[0].split('.')[-1]
        api = api.replace('-', ' ')
        api_seq_formatted += (api + ' ')
    return api_seq_formatted


def tokenize(text):
    if isinstance(text, str):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)
        tokens = [tok.lower() for tok in tokens]
        tokens = list(filter(lambda x: x not in java_keywords, tokens))
        tokens = list(filter(lambda x: x not in stop_words, tokens))
    else:
        tokens = []
    return tokens


def combine_smy_des(summary, description):
    if not isinstance(summary, str):
        summary = ''
    if not isinstance(description, str):
        description = ''

    smy = tokenize(summary)
    des = tokenize(description)
    return smy + des


def parse_specific_code_revision(project_id, commit_id, local_path, url):
    data = {
        "projectID": project_id,
        "commitId": commit_id}
    headers = {
        "content-type": "application/json"}

    response = requests.post(
        url=url,
        headers=headers,
        data=json.dumps(data))

    content = response.content.decode('utf-8')

    content = json.loads(content)

    if 'result' not in content:
        with open(local_path, 'w') as f:
            f.write(json.dumps(content['methodMap']))
    else:
        return content['message']


def process_method_uri(method_uri_raw, method_pos):
    method_pos = method_pos.split('.java')[0].replace('/', '.')
    import re
    p1 = re.compile(r'[(](.*?)[)]', re.S)  # 最小匹配
    method_paras = re.findall(p1, method_uri_raw)[0]
    method_paras_split = method_paras.split(',')  # 方法的所有参数
    if method_paras_split[0] != '':  # 该方法有参数
        paras_short = ''
        for p in method_paras_split:
            para_type = p.split('.')[-1]
            paras_short = paras_short + para_type + ','
        paras_short = paras_short.rstrip(',')
        method_uri = method_uri_raw.replace(method_paras, paras_short)
    else:
        method_uri = method_uri_raw
    method_signature = method_uri.split('-')[1]
    method_ = method_pos + '-' + method_signature

    return method_


def process_json_file(local_json_path, project_name):
    with open(local_json_path, 'r', encoding='utf-8')as f:
        lines = f.read()
    # dict_keys(['classList', 'methodMap', 'methodSubGraphMap', 'variableMap'])
    json_file = json.loads(lines)
    # variable_map=json_file['variableMap']
    method_map = json_file['methodMap']
    # class_list = json_file['classList']
    methodsubgraphmap = json_file['methodSubGraphMap']
    method_dict = {}
    for index, value in method_map.items():  # methodMap 存了每一个类
        for method in value:  # value 是list 里面存了每一个方法的信息
            # labels = method['labels']
            properties = method['propertys']
            method_uri_raw = properties['uri'].split('#')[0]
            method_pos = properties['position']
            # swt的method_pos多了一个版本号出来 在项目名称后面
            # swt 特殊处理
            if project_name == 'SWT':
                method_pos_list = method_pos.split('/')
                method_pos = '.'.join(method_pos_list[i] for i in range(2, len(method_pos_list)))
                method_pos = method_pos_list[0] + '.' + method_pos

            method_uri = process_method_uri(method_uri_raw, method_pos)
            source_code = method['propertys']['sourceCode']
            method_dict[method_uri] = source_code

    method_df = pd.DataFrame.from_dict(method_dict, orient='index', columns=['code'])
    return method_df


def extract_method_structured_features(path):
    def format_method_uri(pos, raw_uri):
        p1 = re.compile(r'[(](.*?)[)]', re.S)  # min match
        paras = re.findall(p1, raw_uri)[0]
        method_paras_split = paras.split(',')  # parameters
        if method_paras_split[0] != '':  # has parameters
            paras_short = ''
            for p in method_paras_split:
                para_type = p.split('.')[-1]
                paras_short = paras_short + para_type + ','
            paras_short = paras_short.rstrip(',')
            uri = raw_uri.replace(paras, paras_short)
        else:
            uri = raw_uri
        method_signature = uri.split('-')[1]
        fixed_uri = pos + '-' + method_signature
        return fixed_uri

    with open(path, 'r', encoding='utf-8')as f:
        methods = json.loads(f.read())

    structured_features = []
    for index, value in methods.items():  # methodMap 存了每一个类
        for method in value:  # value 是list 里面存了每一个方法的信息
            properties = method['propertys']
            uri = properties['uri'].split('#')[0]  # remove version info
            position = properties['position'].split('.java')[0].replace('/', '.')
            method_uri = format_method_uri(position, uri)
            source_code = properties['sourceCode']
            num_statements = source_code.count(';')

            if 'comment' in properties:
                comment = properties['comment']
            else:
                comment = ''

            if 'apiseq' in properties:
                api_seq = properties['apiseq']
            else:
                api_seq = ''

            structured_features.append([method_uri,source_code, api_seq, comment,  num_statements])

    features = pd.DataFrame(structured_features, columns=['uri','tokens', 'api', 'comment', 'num_statements'])
    features.set_index(['uri'],inplace=True)
    features['tokens'] = features['tokens'].apply(tokenize)
    features['comment'] = features['comment'].apply(tokenize)
    features['api'] = features['api'].apply(format_api_seq)
    return features



def stem_tokens(tokens):
    stemmer = PorterStemmer()
    # removed_stopwords = [stemmer.stem(item) for item in tokens if item not in stopwords.words("english")]
    # return removed_stopwords
    stemmed_tokens = [stemmer.stem(item) for item in tokens]
    return stemmed_tokens


def normalize(text):
    remove_punc_map = dict((ord(char), None) for char in string.punctuation)
    removed_punc = text.lower().translate(remove_punc_map)
    tokenized = word_tokenize(removed_punc)
    stemmed_tokens = stem_tokens(tokenized)

    return stemmed_tokens


def cosine_sim(text1, text2):
    vectorizer = TfidfVectorizer(tokenizer=normalize, min_df=1, stop_words="english")
    tfidf = vectorizer.fit_transform([text1, text2])
    sim = (tfidf * tfidf.T).A[0, 1]

    return sim


def top_k_wrong_files(right_files, br_raw_text, java_files, k=20):
    # Randomly sample 2*k files
    randomly_sampled = random.sample(set(java_files.index.tolist()) - set(right_files), 2 * k)

    all_files = []
    for filename in randomly_sampled:
        src = java_files[java_files.index == filename]['code'].values[0]
        rvsm = cosine_sim(br_raw_text[0], src)
        all_files.append((filename, rvsm))
    top_k_files = sorted(all_files, key=lambda x: x[1], reverse=True)[:k]
    return top_k_files


def get_months_between(d1, d2):
    d1 = pd.Timestamp(d1)  # numpy.datetime64 to pd.Timestamp
    diff_in_months = abs((d1.year - d2.year) * 12 + d1.month - d2.month)
    return diff_in_months


def get_previous_fixed_reports(method, until, bug_reports):
    def find_method_occurence(methods):
        if method in methods:
            return 1
        else:
            return 0

    pre_reports = bug_reports[bug_reports['commit_time'] < until]
    pre_reports['fre'] = pre_reports['method'].apply(find_method_occurence)
    pre_reports = pre_reports[pre_reports['fre'] == 1]
    return pre_reports


def most_recent_report(reports):
    if len(reports):
        reports = reports.sort_values("commit_time")
        return reports.tail(1)
    return None


def bug_fixing_recency(commit_time, prev_fixed_reports):
    mrr = most_recent_report(prev_fixed_reports)
    if mrr is not None:
        for index, row in mrr.iterrows():
            mrr_report_time = row['commit_time']
        return 1 / float(get_months_between(commit_time, mrr_report_time) + 1)
    return 0


def calculate_bfr_and_bff(method, commit_time, prev_reports):
    # Bug Fixing Frequency
    prev_fixed_reports = get_previous_fixed_reports(method, commit_time, prev_reports)
    bff = len(prev_fixed_reports)

    # Bug Fixing Recency
    bfr = bug_fixing_recency(commit_time, prev_fixed_reports)

    return float(bfr), float(bff)


def collaborative_filtering_score(cur_report, prev_reports, k=50,
                                  whether_expand_commit=True, commit2commit=None,
                                  whether_expand_method=True, method2method=None):
    """
        给定一个bug report计算其之前出现的所有相似度高的bug reports
        然后计算该bug report与相似bug report所修改过的方法的与该bug report的协同过滤分数
    :param cur_report: 当前bug report
    :param prev_reports: 当前report 按时间排序其之前的所有bug report
    :param commit2commit: commit之间的相似度
    :param method2method: method之间的相似度
    :param k: 取前k个相似度高的commit
    :param whether_expand_commit:是否扩充相似commit
    :param whether_expand_method:是否扩充相似method
    :return: 协同过滤分值高的方法的dict
    """
    sim_commits = {}
    for index, row in prev_reports.iterrows():
        report = row["report"]
        commit_id = row['commit_id']
        sim_score = cosine_sim(' '.join(cur_report), ' '.join(report))
        sim_commits[commit_id] = sim_score

    # 按bug report的 cos_sim 进行排序取前k个
    sim_commits_sorted_top_k = sorted(sim_commits.items(), key=lambda item: (item[1], item[0]), reverse=True)[:k]
    sim_commits_ids = [i[0] for i in sim_commits_sorted_top_k]

    all_ids = list(sim_commits)
    for key in all_ids:
        if key not in sim_commits_ids:
            sim_commits.pop(key)

    # 对所得的相似commit集合进行扩充
    if whether_expand_commit:
        expand_commits = {}
        for commit_id in sim_commits_ids:
            #  找出所有与当前sim commit 相似的 commit
            commits = commit2commit[(commit2commit['commit1'] == commit_id) | (commit2commit['commit2'] == commit_id)]
            length = len(commits)
            for index, row in commits.iterrows():
                sim_commit_id = row['commit1'] if row['commit1'] != commit_id else row['commit2']
                score = row['sim_score']
                if sim_commit_id not in expand_commits:
                    expand_commits[sim_commit_id] = 0
                expand_commits[sim_commit_id] += (score / length)
        # 对扩充的commit也取前k个
        expand_commits_sorted_top_k = sorted(expand_commits.items(), key=lambda item: (item[1], item[0]), reverse=True)[
                                      :k]
        expand_commits_ids = [i[0] for i in expand_commits_sorted_top_k]
        for commit_id in expand_commits_ids:
            if commit_id in sim_commits:
                sim_commits[commit_id] += expand_commits[commit_id]
            else:
                sim_commits[commit_id] = expand_commits[commit_id]

    method_cfs = {}
    # 遍历sim_bug_report计算sim_method的协同过滤分数
    for commit_id, score in sim_commits.items():
        cur_sim_commit = prev_reports[prev_reports['commit_id'] == commit_id]
        if len(cur_sim_commit) == 0:  # 当前commit可能不可见
            continue
        sim_commit_methods = eval(cur_sim_commit['method'].values[0])
        num_methods = len(sim_commit_methods)
        for method in sim_commit_methods:
            if method not in method_cfs:
                method_cfs[method] = 0
            method_cfs[method] += (sim_commits[commit_id] / num_methods)

    if whether_expand_method:
        for method_uri in list(method_cfs.keys()):
            #  找出所有与当前sim commit 相似的 commit
            sim_methods = method2method[
                (method2method['method1'] == method_uri) | (method2method['method2'] == method_uri)]
            length = len(sim_methods)
            for index, row in sim_methods.iterrows():
                sim_method_uri = row['method1'] if row['method1'] != method_uri else row['method2']
                score = row['sim_score']
                if sim_method_uri not in method_cfs:
                    method_cfs[sim_method_uri] = 0
                method_cfs[sim_method_uri] += (score / length)

    return method_cfs


def get_relevant_methods(cur_method_uri, method2method, method_call_method, method_call_graph,
                         k=10):
    """
     为当前短方法找到相关的的方法来扩充，相关关系有 方法相似度 方法调用关系 方法共现调用分数
    :param k:
    :param cur_method_uri: 当前方法uri
    :param method2method:  方法相似度
    :param method_call_method: 方法共现调用分数 A->B , A->C score(B,C)
    :param method_call_graph: 方法调用关系
    :return: 用于扩充的方法uri集合
    """

    def func(current_method_uri, dataframe, by_col):
        ret = {}
        sim_methods = dataframe[
            (dataframe['method1'] == current_method_uri) | (dataframe['method2'] == current_method_uri)]
        for index, row in sim_methods.iterrows():
            sim_m = row['method1'] if row['method1'] != cur_method_uri else row['method2']
            ret[sim_m] = row[by_col] if by_col is not None else -1
        ret = sorted(ret.items(), key=lambda x: x[1], reverse=True)
        ret = [i[0] for i in ret]
        return ret

    expand_methods = set()
    if method_call_graph is None:
        k += 10
    if method_call_method is None:
        k += 10

    methods_1 = func(cur_method_uri, method2method, 'sim_score')

    for method in methods_1[:k]:
        expand_methods.add(method)

    if method_call_method is not None:
        methods_2 = func(cur_method_uri, method_call_method, 'call_score')
        for method in methods_2[:k]:
            expand_methods.add(method)

    if method_call_graph is not None:
        methods_3 = func(cur_method_uri, method_call_graph, None)
        for method in methods_3[:k]:
            expand_methods.add(method)

    if cur_method_uri in expand_methods:
        expand_methods.remove(cur_method_uri)

    return expand_methods


def load_sim_relations(sim_file_path, project_name):
    # similarity between commits

    commit2commit = pd.read_csv(os.path.join(sim_file_path, 'commit2commit.txt'), header=None, sep='\\s+',
                                usecols=[0, 1, 4],
                                names=['id1', 'id2', 'sim_score'])

    # map of (id, commit_id) for commits
    commit_id_map_path = os.path.join(sim_file_path + 'commitIdMap.txt')
    if project_name=='birt'or project_name=='jdt':
        commit_id_map = pd.read_csv(commit_id_map_path, header=None, sep='\\@',
                                    names=['commit_id','id'])
    else:
        commit_id_map = pd.read_csv(commit_id_map_path, header=None, sep='\\s+',
                                names=['id', 'commit_id'])

    # similarity between methods
    method2method = pd.read_csv(os.path.join(sim_file_path, 'method2method.txt'), header=None, sep='\\s+',
                                usecols=[0, 1, 4],
                                names=['id1', 'id2', 'sim_score'])

    # map of (id, method_uri) for methods
    if project_name == 'swt':
        with open(os.path.join(sim_file_path, 'methodIdMap.txt')) as f:
            lines = f.readlines()
        method_id = []
        method_uri =[]
        for line in lines:
            seps = line.split(' ', 1)
            mid = int(seps[0])
            uri = seps[1].rstrip('\n')
            method_id.append(mid)
            method_uri.append(uri)
        method_id_map = pd.DataFrame(columns=['id','method_uri'])
        method_id_map['id'] = method_id
        method_id_map['method_uri'] = method_uri
    elif project_name=='jdt' or project_name == 'birt':
        method_id_map = pd.read_csv(os.path.join(sim_file_path, 'methodIdMap.txt'), header=None, sep='\\@',
                                    names=['method_uri','id'])
    else:
        method_id_map = pd.read_csv(os.path.join(sim_file_path, 'methodIdMap.txt'), header=None, sep='\\s+',
                                names=['id', 'method_uri'])

    # id to commit id in commit2commit
    commit_id_map = commit_id_map.set_index('id').to_dict()['commit_id']
    id2commit = lambda x: commit_id_map[x]
    commit2commit['commit1'] = commit2commit['id1'].apply(id2commit)
    commit2commit['commit2'] = commit2commit['id2'].apply(id2commit)
    commit2commit.drop(['id1', 'id2'], axis=1, inplace=True)

    # id to method_uri in method2method method_call_method method_call_graph
    func = lambda method_uri: method_uri.replace('/', '.')
    method_id_map['method_uri'] = method_id_map['method_uri'].apply(func)
    method_id_map = method_id_map.set_index('id').to_dict()['method_uri']
    id2method = lambda x: method_id_map[x] if x in method_id_map else ''
    method2method['method1'] = method2method['id1'].apply(id2method)
    method2method = method2method[method2method['method1'] != '']
    method2method['method2'] = method2method['id2'].apply(id2method)
    method2method.drop(['id1', 'id2'], axis=1, inplace=True)
    method2method = method2method[method2method['method2'] != '']

    # call relation between methods
    method_call_path = os.path.join(sim_file_path, 'methodCallmethod.txt')
    if os.path.exists(method_call_path):
        method_call_method = pd.read_csv(method_call_path, header=None, sep='\\s+',
                                         usecols=[0, 1, 4],
                                         names=['id1', 'id2', 'call_score', ])

        method_call_graph = pd.read_csv(os.path.join(sim_file_path, 'methodCallGraph.txt'), header=None, sep='\\s+',
                                        names=['id1', 'id2', ])
        method_call_method['method1'] = method_call_method['id1'].apply(id2method)
        method_call_method['method2'] = method_call_method['id2'].apply(id2method)
        method_call_method.drop(['id1', 'id2'], axis=1, inplace=True)

        method_call_graph['method1'] = method_call_graph['id1'].apply(id2method)
        method_call_graph['method2'] = method_call_graph['id2'].apply(id2method)
        method_call_graph.drop(['id1', 'id2'], axis=1, inplace=True)
    else:
        method_call_method, method_call_graph = None,None
    return commit2commit, method2method, method_call_method, method_call_graph
