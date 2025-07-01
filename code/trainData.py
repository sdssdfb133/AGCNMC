from __future__ import division


class Dataset(object):
    def __init__(self, opt, dataset):
        self.data_set = dataset # 设置data_set就为传入的dataset

    def __getitem__(self, index):
        return (self.data_set['dd'], self.data_set['mm'],
                self.data_set['md'], None,
                self.data_set['md_train'], self.data_set['md_true'])

    def __len__(self):
        return self.nums



