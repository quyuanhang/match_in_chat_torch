import numpy as np
from networks.torchMatchModel import MatchModel
from tqdm import tqdm
import torch
from sklearn import metrics


def feed_dict(data, model: MatchModel):
    jds, cvs, labels = data
    inf_mask, zero_mask = model.get_masks(jds, cvs)
    fd = {
        'jd': torch.LongTensor(jds),
        'cv': torch.LongTensor(cvs),
        'inf_mask': torch.FloatTensor(inf_mask),
        'zero_mask': torch.FloatTensor(zero_mask),
        'label': torch.LongTensor(labels)
    }
    if torch.cuda.is_available():
        fd = {k: v.cuda() for k, v in fd.items()}
    return fd


def train(model, train_generator, test_generator, lr=0.0005, n_epoch=100):
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )
    criterion = torch.nn.BCELoss()
    for epoch in range(n_epoch):
        epoch_loss, epoch_metric, valid_metrics = [], [], []
        for i, batch in tqdm(list(enumerate(train_generator))):
            fd = feed_dict(batch, model)
            batch_predict = model.forward(
                fd['jd'], fd['cv'], fd['inf_mask'], fd['zero_mask'])
            batch_loss = criterion(batch_predict, fd['label'].float().view(-1, 1))
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_loss.append(batch_loss.item())
            batch_metric = metrics.roc_auc_score(
                fd['label'].cpu().numpy(),
                batch_predict.cpu().detach().numpy())
            epoch_metric.append(batch_metric)

        for i, batch in tqdm(list(enumerate(test_generator))):
            fd = feed_dict(batch, model)
            batch_predict = model.forward(
                fd['jd'], fd['cv'], fd['inf_mask'], fd['zero_mask'])
            batch_metric = metrics.roc_auc_score(
                fd['label'].cpu().numpy(),
                batch_predict.cpu().detach().numpy())
            valid_metrics.append(batch_metric)
        print('epoch: {}, train loss: {:.3f}, train metric: {:.3f}, valid metric: {:.3f}'.format(
            epoch,
            np.array(epoch_loss).mean(),
            np.array(epoch_metric).mean(),
            np.array(valid_metrics).mean()))

def valid(test_generator, model):
    valid_metrics = []
    for i, batch in tqdm(list(enumerate(test_generator))):
        fd = feed_dict(batch, model)
        batch_predict = model.forward(
            fd['jd'], fd['cv'], fd['inf_mask'], fd['zero_mask'])
        batch_metric = metrics.roc_auc_score(
            fd['label'].cpu().numpy(),
            batch_predict.cpu().detach().numpy())
        valid_metrics.append(batch_metric)
    print('valid metric: {:.3f}'.format(np.array(valid_metrics).mean()))





