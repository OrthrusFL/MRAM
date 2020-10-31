import os
import logging
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
from utils import parse_specific_code_revision


def parse_code_revision(conf):
    logger = logging.getLogger(conf['log'])
    repo = pd.read_csv(conf['report']).sort_values(by='commit_time', ascending=True)
    logger.info('len of the reports:{}'.format(len(repo)))

    def func(row):
        commit_id = row['commit_id'].values[0]
        json_save_path = os.path.join(conf['json_dir'], commit_id + '.json')
        if not os.path.exists(json_save_path):
            project_id = conf['target']
            if project_id == 'aspectj':
                project_id = 'org.aspectj'
            elif project_id == 'swt':
                project_id = 'eclipse.platform.swt'
            elif project_id == 'jdt':
                project_id = 'eclipse.jdt.ui'
            else:
                pass
            msg = parse_specific_code_revision(project_id=project_id,
                                               commit_id=commit_id,
                                               local_path=json_save_path,
                                               url=conf['URL'])
            if msg is not None:
                logger.info('parse fails:{} (error msg:{})'.format(commit_id, msg))
            else:
                logger.info('parse success:{}!'.format(commit_id))

    # group by and parallel processing
    repo_grouped = repo.groupby(repo.index)
    Parallel(n_jobs=cpu_count()-6)(delayed(func)(row) for index, row in tqdm(repo_grouped))

    # single process
    # for index, row in repo.iterrows():
    #    func(row)
    logger.info('finished!')
