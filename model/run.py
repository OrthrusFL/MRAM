import argparse
import logging
from configs import config
from vocab import build_vocab
from parse import parse_code_revision
from prepare import prepare_data
from predict import within_project_prediction, cross_project_prediction


def parse_args():
    parser = argparse.ArgumentParser("Locating faulty methods")

    parser.add_argument('--project', required=True,
                        choices=['swt', 'tomcat', 'aspectj', 'jdt', 'birt'],
                        help='specific the target project')

    parser.add_argument('--parse', action='store_true',
                        help='parse specific code revision')
    parser.add_argument('--prepare', action='store_true',
                        help='prepare training data and testing data')
    parser.add_argument('--vocab', action='store_true',
                        help='build vocabulary')
    parser.add_argument('--within', action='store_true',
                        help='within project prediction')
    parser.add_argument('--cross', choices=['swt', 'tomcat', 'aspectj', 'jdt', 'birt'],
                        help='cross project prediction (the para here indicates the source project)')

    parser.add_argument('--bias', action='store_true', help='whether to use localized bug reports')
    parser.add_argument('--log', type=str, default='MFL', help='specific file to record log')
    parser.add_argument('--gpu', type=str, default=None, help='specify gpu device')

    return parser.parse_args()


def run():
    args = parse_args()
    conf = config(args)

    if args.log:
        conf['log'] = args.log
        conf['log_dir'] = './' + args.log + '.log'

        logger = logging.getLogger(conf['log'])
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(conf['log_dir'])
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info('---------------------------------------------------------------------------------------')
        logger.info('Running with args : {}'.format(args))

    if args.vocab:
        build_vocab(conf)

    if args.parse:
        parse_code_revision(conf)

    if args.prepare:
        prepare_data(conf)

    if args.within:
        within_project_prediction(conf)

    if args.cross:
        cross_project_prediction(conf)


if __name__ == '__main__':
    run()
