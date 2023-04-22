class Accuracy(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, num, n):
        self.sum += num
        self.count += n
        self.avg = self.sum / self.count