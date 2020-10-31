import os


def build_dir():
    output_dir = os.path.dirname('./output/')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for project in ['aspectj', 'birt', 'jdt', 'swt', 'tomcat']:

        project_dir = os.path.join(output_dir, project)
        if not os.path.exists(project_dir):
            os.mkdir(project_dir)

        # path to save model
        model_dir = os.path.join(project_dir, 'model/')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        # path to save prepared data
        data_dir = os.path.join(project_dir, 'data/')
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        # path to save extracted methods from multiple code revisions
        json_dir = os.path.join(project_dir, 'json/')
        if not os.path.exists(json_dir):
            os.mkdir(json_dir)

        # path to load sim relations from revision analyzer
        sim_dir = os.path.join(project_dir, 'sim/')
        if not os.path.exists(sim_dir):
            os.mkdir(sim_dir)


def config(args):
    # mkdir
    build_dir()

    conf = {
        # specific the URL of interface: version_info (from code revision graphs)
        'URL': 'http://202.120.40.28:5689/parser/KGWeb/versionInfo',
        'log': '',
        'log_dir': '',
        'source': args.cross,
        'target': args.project,
        'bias':args.bias,
        'report_dir': os.path.dirname('./report/'),
        'output_dir': os.path.dirname('./output/'),
        'vocab_dir': os.path.dirname('./output/vocab/'),
        'k_fold': 10,

        # model parameters
        'emb_size': 512,
        'min_count': 3,
        'hidden_size': 512,
        'max_len': 2000,
        'epoch':1,
        'batch_size': 64,
        'gpu': args.gpu,
        'augment_threshold': 5

    }

    conf['project_dir'] = os.path.join(conf['output_dir'], conf['target'])
    conf['data_dir'] = os.path.join(conf['project_dir'], 'data/')
    conf['model_dir'] = os.path.join(conf['project_dir'], 'model/')
    conf['json_dir'] = os.path.join(conf['project_dir'], 'json/')
    conf['sim_dir'] = os.path.join(conf['project_dir'], 'sim/')
    conf['report'] = os.path.join(conf['report_dir'], args.project + '.csv')

    return conf
