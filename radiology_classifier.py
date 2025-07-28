import requests
import json
import numpy as np
import hdbscan
from tqdm import tqdm
from collections import defaultdict
import datetime
import pandas as pd
import sqlalchemy as sql
import pandas as pd
from bs4 import BeautifulSoup
import re
from json_repair import repair_json
import logging
import logging.config
import multiprocessing
import os
from functools import lru_cache
import time
from keyword_extraction import get_orientation_position,drop_dict_duplicates
from dotenv import load_dotenv
load_dotenv()
DEFAUT_DATA = os.getenv('DEFAUT_DATA')
DEFAUT_CASE= os.getenv('DEFAUT_CASE')
ENDPOINT_IP=os.getenv('ENDPOINT_IP')
# 日志配置字典
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{asctime} - {levelname} - {module}:{lineno} - {message}',
            'style': '{',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose'
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'cllasifier_radiology.log',
            'maxBytes': 100 * 1024 * 1024,  # 100MB
            'backupCount': 5,
            'formatter': 'verbose',
            'encoding': 'utf-8'
        }
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO'
    }
}

# 配置日志d
logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)
connectionString = 'mssql+pyodbc://RIS:RIS@integration'
inteEngine = sql.create_engine(connectionString)
# --- 1. 配置区域 ---
#进程数量
n_processes=16
# LLM API 配置 (用于实体抽取和类别命名)
LLM_API_URL = os.getenv('LLM_API_URL')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME')

# Embedding API 配置
EMBEDDING_API_URL = os.getenv('EMBEDDING_API_URL')
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
ALL_TOKENS=0
# --- 2. 辅助函数与API封装 ---

@lru_cache(maxsize=5000)
def call_llm_api(prompt:str, model_name:str):
    """
    调用本地部署的LLM API。
    参数:
        prompt (str): 发送给模型的提示词。
        model_name (str): 要使用的模型名称。
    返回:
        str: 模型返回的文本内容。
    """
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, # 对于抽取和命名任务，使用低temperature以保证稳定性
        "max_tokens":4096,
        "stream": False
    }
    try:
        response = requests.post(LLM_API_URL, headers=headers, data=json.dumps(payload), timeout=240)
        response.raise_for_status()
        parseStr= response.json()["choices"][0]["message"]["content"]
        ALL_TOKENS=+ response.json()['usage']['total_tokens']
        parseStr=re.sub(r'<think>.*?</think>', '', parseStr.replace("\n"," "),flags=re.DOTALL | re.MULTILINE |re.I)
        if "命名" not in prompt:
            mat=re.search(r"\s\[.*?\]",parseStr,re.DOTALL | re.MULTILINE |re.I)
            if mat:
                resultJson=mat.group(0)
            else:
                resultJson=re.sub("json|\r|\n|`", "", parseStr)
            resultJson = repair_json(resultJson)
            return resultJson
        else:
            if len(parseStr)>50:
                parseStr=re.sub(r'<think>.*?</think>', '', parseStr.replace("\n"," "),flags=re.DOTALL | re.MULTILINE |re.I)
                matches=re.findall(r"\*\*(.*?)\*\*",parseStr,flags=re.I)
                if len(parseStr)>50 or len(matches)==0:
                    logger.error("处理失败："+parseStr)
                    return ''
                else:
                    return ",".join(matches)
            return parseStr
    except requests.exceptions.RequestException as e:
        logger.error(f"调用LLM API时发生网络错误: {e}")
        return None


def call_embedding_api(text_list:list, model_name:str):
    """
    调用本地部署的Embedding API。
    参数:
        text_list (list of str): 需要生成嵌入的文本列表。
        model_name (str): 要使用的模型名称。
    返回:
        list of list of float: 文本对应的嵌入向量列表。
    """
  

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "input": text_list
    }
    try:
        response = requests.post(EMBEDDING_API_URL, headers=headers, data=json.dumps(payload), timeout=120)
        response.raise_for_status()
        # 提取嵌入向量
        embeddings = [item['embedding'] for item in response.json()['data']]
        return embeddings
    except requests.exceptions.RequestException as e:
        logger.error(f"调用Embedding API时发生网络错误: {e}")
        return None

@lru_cache(maxsize=5000)
def build_extraction_prompt(report_text:str):
    """
    构建用于实体抽取的提示词。
    """

    prompt="""
你是一名专业的放射科医生，你的任务是从一份放射诊断报告中精准、完整地抽取出所有的诊断。/no_think

【要求】
1.  **全面抽取**：一份报告可能包含多个独立的诊断结论（例如，一个主要诊断和一个次要诊断，或多个部位的共同诊断），请务必全部找出。
2.  **精确提取**：只提取核心的、最精炼的诊断名称，例如“腺癌”、“磨玻璃结节”、“骨折”。忽略描述性文字(如大小、形态、轻重、形成、建议、可疑、除外)，忽略与历史进行比较的情况（如增大、缩小、好转）。
3.  **格式严格**：必须以JSON格式的列表(List of Objects)返回，每个Object包含"诊断"字段。
4.  **无诊断情况**：如果报告中没有明确的诊断结论（例如“导管头部位于第4胸椎上缘”），请返回一个空列表 `[]`。

【示例】
报告原文：“左肺下叶、右肺上叶各见一枚磨玻璃结节，大小3-5mm。”
JSON输出：
[
    {"诊断": ["磨玻璃结节"]}
]

报告原文：“心肺膈未见异常”
JSON输出：
[
    {"诊断": ["正常"]}
]


【待处理报告】
报告原文："{report_text}"
JSON输出：
        
        """
    prompt=prompt.replace("{report_text}", report_text)
    return prompt


def build_naming_prompt(diagnosis_list):
    """
    构建用于类别命名的提示词。
    """
    # 为了防止prompt过长，对样本进行采样
    if len(diagnosis_list) > 20:
        samples = np.random.choice(diagnosis_list, 20, replace=False).tolist()
    else:
        samples = diagnosis_list
    samples_text= json.dumps(samples, ensure_ascii=False, indent=2)   

    prompt="""
你是一位资深的放射学专家。请分析以下从多份放射报告中提取的、描述同一种疾病的词条列表。
你的任务是：
1.  归纳总结这些词条的核心诊断。
2.  生成一个最精准、最通用、最标准的“官方诊断名称”作为该类别的统一命名。
3.  你的统一命名只能使用中文，不要英文名
4.  你的命名必须是最精简，字数最少，不要带注释，不要详细解释，例如“骨质增生、肥厚”，请返回“骨质增生”，例如“心脏支架术后改变”，请返回“支架术后”

【要求】
- 返回的名称应尽可能简练且具有医学权威性。
- 请只返回最终的诊断名称，不要任何解释或多余的文字。

【词条列表】
{samples_text}

【标准诊断名称】
    """
    prompt=prompt.replace("{samples_text}", samples_text)
    return prompt

# --- 3. 核心处理流程 ---
def clean_html(text):
    """清洗HTML标签"""
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    return cleaned_text
def step0_get_data(start_time:str,end_time:str):
    """获取数据"""
    sqlstr = """select AccessionNumber as 影像号,ModalityType as 设备,ProcedureDesc as 部位,Impression as 结论 
        from tAllReportInfo where  StudyDateTime between '%s 00:00:00' and '%s 23:59:59'  
        and (ModalityType='CT' or ModalityType='MR' or ModalityType='DR' or ModalityType='MG') and Representation is not NULL 
        order by StudyDateTime""" %(start_time,end_time)
    QueryResult= pd.read_sql(sqlstr, inteEngine)
    logger.info(f"--- 步骤 0: 获取了{len(QueryResult)}份报告")
    QueryResult['结论']=QueryResult['结论'].apply(lambda x:clean_html(x))
    args = list(QueryResult[["部位","结论","设备","影像号"]].itertuples(index=False,name=None))
    with multiprocessing.Pool(n_processes) as pool:
        # 使用reportNLP执行部位实体抽取任务
        results = list(tqdm(pool.imap_unordered(report_analysis, args), total=len(args), desc="NLP部位抽取"))
    
    return [item for sublist in results for item in sublist]

def report_analysis(ReportTxt):
    """将报告的描述与结论字段的自然语言转换为结构化字段，以便存入数据库中.
    输出格式：partlist：六级部位列表；position：标准化部位名称；word：原始部位名称；
    illness：疾病描述；measure：轴测量值；percent：百分比值；primary：预处理后的报告语句
    """
    StudyPart,ConclusionStr,Modality,accno=ReportTxt
    studypart_analyze = get_orientation_position(StudyPart, title=True) if StudyPart else []
    Conclusion_analyze = get_orientation_position(ConclusionStr, add_info=[
        s['axis'] for s in studypart_analyze]) if ConclusionStr else []

    if len(Conclusion_analyze)>0:
        Conclusion_analyze=[{**dic, 'Modality': Modality} for dic in Conclusion_analyze]
        Conclusion_analyze=[{**dic, 'accno': accno} for dic in Conclusion_analyze]
    return Conclusion_analyze

@lru_cache(maxsize=5000)
def worker_extract_entities_with_report(report:str):
    """多进程工作函数：处理单个报告的实体抽取，并返回报告原文和抽取结果"""
    prompt = build_extraction_prompt(report)
    response_text = call_llm_api(prompt, LLM_MODEL_NAME)
    
    if response_text:
        try:
            extracted_data = json.loads(response_text)
            if isinstance(extracted_data, list):
                # 返回报告原文和有效的诊断名称列表
                diagnoses = [item["诊断"] for item in extracted_data if "诊断" in item and item["诊断"]]
                return report, [item for sublist in diagnoses for item in sublist]
        except json.JSONDecodeError:
            logger.error(f"[警告] LLM返回的不是有效的JSON格式，已跳过。返回内容: {response_text}")
            return report, None
    return report, None

def step2_generate_embeddings(diagnoses):
    """
    步骤二：为所有诊断实体生成嵌入向量。
    """
    logger.info("--- 步骤 2: 开始批量生成嵌入向量 ---")
    # Embedding API通常支持批量处理，但这里为简单起见，可以分小批次处理
    batch_size = 32
    all_embeddings = []
    for i in tqdm(range(0, len(diagnoses), batch_size), desc="嵌入生成进度"):
        batch_texts = diagnoses[i:i + batch_size]
        batch_embeddings = call_embedding_api(batch_texts, EMBEDDING_MODEL_NAME)
        if batch_embeddings:
            all_embeddings.extend(batch_embeddings)
            
    logger.info(f"--- 步骤 2 完成: 成功生成 {len(all_embeddings)} 个嵌入向量 ---")
    return np.array(all_embeddings)

def step3_cluster_diagnoses(embeddings):
    """
    步骤三：使用HDBSCAN对嵌入向量进行聚类。
    """
    logger.info("--- 步骤 3: 开始使用HDBSCAN进行聚类 ---")
    min_samples_for_a_cluster=10
    merge_distance_threshold=0.2
    # min_cluster_size 是一个重要参数，代表形成一个独立类别所需要的最少样本数
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_samples_for_a_cluster, 
                                metric='euclidean',
                                cluster_selection_epsilon=merge_distance_threshold,
                                gen_min_span_tree=True,
                                allow_single_cluster=True,
                                core_dist_n_jobs=-1)
    cluster_labels = clusterer.fit_predict(embeddings)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = np.sum(cluster_labels == -1)
    
    logger.info(f"--- 步骤 3 完成: 发现 {n_clusters} 个类别, {n_noise} 个无法聚类点 ---")
    return cluster_labels,n_clusters


@lru_cache(maxsize=5000)
def worker_standard_name(label):
    
    prompt="""
你是一位资深的放射学专家。请分析以下词条。/no_think
你的任务是：
1.  生成一个最精准、最通用、最标准的“官方诊断名称”作为该词条的统一命名。
3.  你的命名只能使用中文，不要英文名
4.  你的命名必须是最精简，字数最少，不要带注释，不要详细解释，例如“骨质增生、肥厚”，请返回“骨质增生”，例如“心脏支架术后改变”，请返回“支架术后”

【要求】
- 返回的名称应尽可能简练且具有医学权威性。
- 请只返回最终的诊断名称，不要任何解释或多余的文字。

【词条】
{label}

【标准诊断名称】
    """
    prompt=prompt.replace("{label}", label)
    standard_name = call_llm_api(prompt, LLM_MODEL_NAME)
    
    if standard_name:
        return standard_name.strip()
    else:
        # 如果API调用失败，返回一个带标签的占位符
        return f"未命名类别_{label}"

def worker_name_cluster(args):
    """多进程工作函数：处理单个类别的命名"""
    label, diagnosis_list = args
    prompt = build_naming_prompt(diagnosis_list)
    canonical_name = call_llm_api(prompt, LLM_MODEL_NAME)
    
    if canonical_name:
        return label, canonical_name.strip()
    else:
        # 如果API调用失败，返回一个带标签的占位符
        return label, f"未命名类别_{label}"

def step4_name_clusters(diagnoses, cluster_labels):
    """
    步骤四：自动为每个聚类命名（多进程版）。
    """
    logger.info("--- 步骤 4: 开始自动为类别命名 (多进程) ---")
    # 按聚类ID对原始诊断文本进行分组
    grouped_diagnoses = defaultdict(list)
    for diagnosis, label in zip(diagnoses, cluster_labels):
        if label != -1: # -1是噪声点，不参与命名
            grouped_diagnoses[label].append(diagnosis)
    
    cluster_names = {}
    
    # 将需要处理的任务打包成列表
    
    tasks = list(grouped_diagnoses.items())
    if tasks:
        with multiprocessing.Pool(n_processes) as pool:
            # 并行执行命名任务
            results = list(tqdm(pool.imap_unordered(worker_name_cluster, tasks), total=len(tasks), desc="类别命名进度"))
            
        # 将返回的(label, name)元组列表转换回字典
        for label, name in results:
            cluster_names[label] = name
            
    #把无法分类的项目添加回去
    base=max(cluster_labels)+1
    for diagnosis, label in zip(diagnoses, cluster_labels):
        if label == -1:
            cluster_names[base] = diagnosis        
            base+=1
    logger.info(f"--- 步骤 4 完成: 成功命名 {len(cluster_names)} 个类别 ---")
    return cluster_names

def step5_build_classifier_index(embeddings, cluster_labels, cluster_names):
    """
    步骤五：计算中心向量并构建最终的分类器索引文件。
    """
    logger.info("\n--- 步骤 5: 开始构建分类器索引 ---")
    classifier_data = []
    
    # 按聚类ID对嵌入向量进行分组
    grouped_embeddings = defaultdict(list)
    for embedding, label in zip(embeddings, cluster_labels):
        if label != -1:
            grouped_embeddings[label].append(embedding)
            
    for label, name in cluster_names.items():
        # 计算中心向量（该类别下所有向量的平均值）
        centroid_vector = np.mean(grouped_embeddings[label], axis=0).tolist()
        classifier_data.append({
            "category_name": name,
            "centroid_vector": centroid_vector
        })
        
    index_file_content = {
        "version": "1.0",
        "created_at": datetime.datetime.now().isoformat(),
        "classifier_data": classifier_data
    }
    
    # 保存到文件
    output_filename = "classifier_index.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(index_file_content, f, ensure_ascii=False, indent=4)
        
    logger.info(f"--- 步骤 5 完成: 分类器索引已保存至 {output_filename} ---")
    return classifier_data

def worker_extract_entities(report):
    """多进程工作函数：处理单个报告的实体抽取"""
    prompt = build_extraction_prompt(report)
    response_text = call_llm_api(prompt, LLM_MODEL_NAME)
    
    if response_text:
        try:
            extracted_data = json.loads(response_text)
            if isinstance(extracted_data, list):
                # 只返回有效的诊断名称列表
                return [item["诊断"] for item in extracted_data if "诊断" in item and item["诊断"]]
        except json.JSONDecodeError:
            # 在工作进程中记录错误，但返回None，主进程会忽略
            logger.error(f"[警告] LLM返回的不是有效的JSON格式，已跳过。返回内容: {response_text}")
            return None
    return None


def process_element(args):
    """
    Process a single (report_text, diagnoses) pair.
    Returns a tuple of (report_text, standardized_diagnoses, diagnoses)
    if diagnoses is non-empty, else returns None.
    """
    report_text, diagnoses = args
    if diagnoses:
        standardized = worker_standard_name(diagnoses)
        return (report_text, standardized, diagnoses)
    else:
        return None


# --- 4. 主执行函数 ---

def main(start_time:str,end_time:str,entities_file=None,embeddings_file=None):
    """
    主函数，串联所有步骤，并增加对原始数据的分布统计功能。
    """
    logger.info("====== 开始执行离线分类体系构建与数据分布统计流程 ======")

    # --- Part 1: 数据获取与分类器构建 ---
    
    pathology_reports = []
    report_to_diagnoses_map = {}
    all_diagnoses = []
    # Step 0: 从PACS数据库获取原始报告数据，实际应用可以改为从文件读取
    pathology_reports = step0_get_data(start_time, end_time)
    logger.info(f"输入: {len(pathology_reports)} 条原始放射报告诊断句子。")
    
    if entities_file is None:
        # 为了节省LLM算力，对报告进行去重，仅对独立报告进行实体抽取
        unique_reports = list(set([x["primary"] for x in pathology_reports]))
        logger.info(f"去重后剩余: {len(unique_reports)} 份诊断句子用于构建分类器。")
        
        # Step 1 : 从独立报告中抽取实体，并建立报告到诊断的映射
        logger.info("--- 步骤 1: 开始从独立报告中批量抽取诊断实体 ---")
        tasks = unique_reports
        with multiprocessing.Pool(n_processes) as pool:
            # 使用 imap_unordered 来提高效率
            results = list(tqdm(pool.imap_unordered(worker_extract_entities_with_report, tasks,chunksize=20), total=len(tasks), desc="实体抽取进度"))
        
        logger.info(f"--- 步骤 1 : 共抽取出 {len(results)} 个实体 ---")

        
        all_diagnoses_flat = []
        tasks = []          # 每个元素是单个诊断字符串
        task_idx = []       # 对应的 report_text，用于结果归类

        for report_text, diagnoses in results:
            if diagnoses:
                for d in diagnoses:
                    tasks.append(d)
                    task_idx.append(report_text)

        # 根据机器实际情况调整进程数
        with multiprocessing.Pool(n_processes) as pool:
            standard_stream = list(tqdm(pool.imap(worker_standard_name, tasks,chunksize=50), total=len(tasks), desc="实体标准化进度"))
            # 组装结果
            all_diagnoses_flat = []
            report_to_diagnoses_map = {}
            for report_text, std_name in zip(task_idx, standard_stream):
                if std_name is None:               # 失败的直接跳过
                    continue
                all_diagnoses_flat.append(std_name)
                report_to_diagnoses_map.setdefault(report_text, []).append(std_name)
        
        # 从所有抽取结果中获取唯一的诊断实体列表
        logger.info(f"--- 步骤 1 : 共获得 {len(all_diagnoses_flat)} 个标准化实体 ---")
        all_diagnoses = list(set(all_diagnoses_flat))
        
        logger.info(f"--- 步骤 1 完成: 共抽取出 {len(all_diagnoses)} 个独立标准化诊断实体，消耗token:{ALL_TOKENS} ---")
        
        if not all_diagnoses:
            logger.error("!!! 未能抽取到任何诊断实体，流程终止。")
            return
        
        # 将抽取的独立实体保存到文件，以备将来重用
        with open("report_to_diagnoses_map.json", "w", encoding='utf-8') as f:
            json.dump(report_to_diagnoses_map, f, ensure_ascii=False)
        logger.info("--- 已保存实体抽取结果到 report_to_diagnoses_map.json")
    else:
        # 如果提供了实体文件，则直接加载，跳过数据获取和抽取步骤
        with open(entities_file, "r", encoding='utf-8') as f:
            report_to_diagnoses_map = json.load(f)
        all_diagnoses=list(set([item for subitem in report_to_diagnoses_map.values() for item in subitem]))
        logger.info(f"--- 从文件 {entities_file} 读取了 {len(all_diagnoses)} 个实体")

    # Step 2: 为所有独立诊断实体生成嵌入向量
    if embeddings_file:
        embeddings = np.load(embeddings_file)
        logger.info(f"--- 已从文件 {embeddings_file} 读取 {len(embeddings)} 个向量")
    else:    
        embeddings = step2_generate_embeddings(all_diagnoses)
        if embeddings.size == 0:
            logger.error("!!! 未能生成任何嵌入向量，流程终止。")
            return
        np.save('all_embeddings.npy', embeddings)
        logger.info("--- 已保存向量结果到 all_embeddings.npy")

    # Step 3: 对嵌入向量进行聚类
    cluster_labels, n_clusters = step3_cluster_diagnoses(embeddings)
    
    # Step 4: 为每个类别自动命名
    cluster_names = step4_name_clusters(all_diagnoses, cluster_labels)
    
    # --- Part 2: 诊断分布统计 (仅当从原始报告开始流程时执行) ---
    if pathology_reports:
        logger.info("\n--- 开始统计原始数据中的诊断分布 ---")
        
        # 建立一个从诊断实体到其标准类别名称的直接映射，方便快速查找
        diagnosis_to_category_name = {}
        noise_category_name = "噪声/未分类" 
        for diagnosis, label in zip(all_diagnoses, cluster_labels):
            # .get(label, ...) 用于处理噪声点 (label = -1)
            diagnosis_to_category_name[diagnosis] = cluster_names.get(label, noise_category_name)

        # 遍历原始报告列表（含重复），利用之前建立的映射进行高效统计
        result_df=[]
        for report in tqdm(pathology_reports, desc="统计诊断分布进度"):
            # 从缓存的映射中获取该报告对应的诊断列表，避免重复抽取
            report['diagnoses'] = report_to_diagnoses_map.get(report['primary'], [])
            result_df.append(report)
        result_df=pd.DataFrame(result_df)
        
        # 将统计结果保存为JSON文件
        output_filename = "diagnosis_distribution.xlsx"
        result_df.to_excel(output_filename,index=False)
        logger.info(f"--- 诊断分布统计完成，结果已保存至 {output_filename} ---")
        
    else:
        logger.info("\n--- 跳过诊断分布统计（因为流程不是从原始报告数据开始） ---")

    # --- Part 3: 构建并保存最终的分类器索引 ---
    final_index = step5_build_classifier_index(embeddings, cluster_labels, cluster_names)
    
    logger.info("====== 离线分类体系构建与统计流程全部完成 ======")
    kinds = list(set([x["category_name"] for x in final_index]))
    logger.info(f"--- 最终生成的分类器包含 {len(kinds)} 个类别。")
    logger.info(kinds)


if __name__ == "__main__":
    start_time='2025-1-1'
    end_time='2025-1-31'
    # 注意：为了执行新增的分布统计功能，请确保主函数调用时
    # 不传入 'entities_file' 或 'embeddings_file' 参数，
    # 以便让脚本从 Step0 开始完整执行。
    # main(start_time, end_time,entities_file="report_to_diagnoses_map.json",embeddings_file="all_embeddings.npy")
    main(start_time, end_time)