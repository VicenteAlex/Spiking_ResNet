import torch


def adjust_learning_rate(optimizer, cur_epoch, max_epoch, d1, d2, d3):
    """ Reduces the learning rate after the predifined epoch numbers"""
    if cur_epoch == (max_epoch*d1) or cur_epoch == (max_epoch*d2) or cur_epoch==(max_epoch*d3):
        print('Decreasing LR by /10')
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10

def recalculate_learning_rate(optimizer, cur_epoch, max_epoch,d1, d2, d3):
    """ Recalculates the value of teh learning rate depending on the epoch"""
    if max_epoch*d1 < cur_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10
    if max_epoch*d2 < cur_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10
    if max_epoch*d3 < cur_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10



def accuracy(outp, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = outp.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




