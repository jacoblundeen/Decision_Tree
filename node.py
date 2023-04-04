class Node:
    '''
    Helper class which implements a single tree node.
    '''
    def __init__(self, feature=None, threshold=None, data_left=None, data_mleft=None, data_middle=None,
                 data_mright=None, data_right=None, gain=None, value=None, parent=None, prediction=None, level=None,
                 node_type=None, samples=None, parent_branch=None, node_num=None):
        self.feature = feature
        self.threshold = threshold
        self.parent = parent
        self.data_left = data_left
        self.data_mleft = data_mleft
        self.data_middle = data_middle
        self.data_mright = data_mright
        self.data_right = data_right
        self.gain = gain
        self.value = value
        self.prediction = prediction
        self.level = level
        self.node_num = node_num
        self.node_type = node_type
        self.samples = samples
        self.parent_branch = parent_branch


