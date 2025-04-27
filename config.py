import configparser


class Config(object):
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print("loading config: %s failed" % (config_file))

        #Hyper-parameter
        self.epochs = conf.getint("Model_Setup", "epochs")
        self.lr = conf.getfloat("Model_Setup", "lr")
        self.weight_decay = conf.getfloat("Model_Setup", "weight_decay")
        self.k = conf.getint("Model_Setup", "k")
        self.nhid1 = conf.getint("Model_Setup", "nhid1")
        self.nhid2 = conf.getint("Model_Setup", "nhid2")
        self.dropout = conf.getfloat("Model_Setup", "dropout")
        self.beta = conf.getfloat("Model_Setup", "beta")
        self.theta = conf.getfloat("Model_Setup", "theta")
        # self.no_cuda = conf.getboolean("Model_Setup", "no_cuda")
        # self.no_seed = conf.getboolean("Model_Setup", "no_seed")
        # self.seed = conf.getint("Model_Setup", "seed")

        # Dataset
        self.n = conf.getint("Data_Setting", "n")
        self.fdim = conf.getint("Data_Setting", "fdim")
        self.class_num = conf.getint("Data_Setting", "class_num")

        # Additional settings
        self.head_num = conf.getint("Additional_Settings", "head_num")
        self.long_walks_per_node = conf.getint("Additional_Settings", "long_walks_per_node")
        self.long_walk_len = conf.getint("Additional_Settings", "long_walk_len")
        self.walk_len = conf.getint("Additional_Settings", "walk_len")
        self.walks_per_node = (1 + self.long_walk_len + 1 - self.walk_len) * self.long_walks_per_node



