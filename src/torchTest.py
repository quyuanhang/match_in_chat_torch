import networks
from utils import dataSet, torchTrainer
import argparse
import os
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default='-1')
    parser.add_argument('--datain', nargs='?', default='interview')
    parser.add_argument('--dataout', default='interview')
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    # word2vec arguments
    parser.add_argument('--w2v', type=int, default=0)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--w2v_ep', type=int, default=1)
    parser.add_argument('--w2v_lr', type=int, default=0.025)
    parser.add_argument('--min_count', type=int, default=5)
    # model arguments
    parser.add_argument('--doc_len', type=int, default=25)
    parser.add_argument('--sent_len', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_epoch', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    # 参数接收器
    args = parse_args()

    # 显卡占用
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    # 预训练word to vector
    if args.w2v:
        word2vec = networks.Word2Vec(
            input_file_name='{}.all'.format(args.datain),
            output_file_name='./data/{}.word_emb'.format(args.dataout),
            emb_dimension=args.emb_dim,
            iteration=args.w2v_ep,
            initial_lr=args.w2v_lr,
            min_count=args.min_count
        )
        word2vec.train()

    # 读取训练
    word_dict = torch.load('data/test/word2id.pkl')
    # with open('data/test/interview.word_emb') as f:
    #     data = f.read().strip().split('\n')
    # embs = np.array(
    #         [
    #         [
    #             int(y) for y in x.split(' ')[1:]
    #         ]
    #         for x in data
    #     ]
    # )

    dataSet.data_split(args.datain, args.dataout, frac=0.1)

    # train_data = dataSet.data_generator(
    #     fp='./data/{}.train'.format(args.dataout),
    #     word_dict=word_dict,
    #     doc_len=args.doc_len,
    #     sent_len=args.sent_len,
    #     batch_size=args.batch_size
    # )

    test_data = dataSet.data_generator(
        fp='./data/{}.test'.format(args.dataout),
        word_dict=word_dict,
        doc_len=args.doc_len,
        sent_len=args.sent_len,
        batch_size=args.batch_size
    )

    model = networks.torchMatchModel.MatchModel(
        n_word=len(word_dict),
        emb_dim=args.emb_dim,
        doc_len=args.doc_len,
        sent_len=args.sent_len,
    )
    for k, v in model.named_parameters():
        print(k)
    model_pre = torch.load('data/test/gj_net_dict.pkl')

    model.jd_cnn.emb_weights.weight.data.copy_(model_pre['G_enc.embedding.weight'])
    model.jd_cnn.cnn.cnn1.weight.data.copy_(model_pre['G_enc.enc.cnn1.0.weight'])
    model.jd_cnn.cnn.cnn1.bias.data.copy_(model_pre['G_enc.enc.cnn1.0.bias'])
    model.jd_cnn.cnn.cnn2.weight.data.copy_(model_pre['G_enc.enc.cnn2.0.weight'])
    model.jd_cnn.cnn.cnn2.bias.data.copy_(model_pre['G_enc.enc.cnn2.0.bias'])
    model.cv_cnn.emb_weights.weight.data.copy_(model_pre['J_enc.embedding.weight'])
    model.cv_cnn.cnn.cnn1.weight.data.copy_(model_pre['J_enc.enc.cnn1.0.weight'])
    model.cv_cnn.cnn.cnn1.bias.data.copy_(model_pre['J_enc.enc.cnn1.0.bias'])
    model.cv_cnn.cnn.cnn2.weight.data.copy_(model_pre['J_enc.enc.cnn2.0.weight'])
    model.cv_cnn.cnn.cnn2.bias.data.copy_(model_pre['J_enc.enc.cnn2.0.bias'])
    model.mlp.bil.weight.data.copy_(model_pre['M_cla.Bil.weight'])
    model.mlp.bil.bias.data.copy_(model_pre['M_cla.Bil.bias'])
    model.mlp.mlp.weight.data.copy_(model_pre['M_cla.MLP.1.weight'])
    model.mlp.mlp.bias.data.copy_(model_pre['M_cla.MLP.1.bias'])

    model = model.cuda()


    torchTrainer.valid(test_data, model)

    print('done')

