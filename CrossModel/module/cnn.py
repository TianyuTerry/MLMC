import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    General module for linear -> cross -> conv -> relu -> maxpool -> cross
    """
    def __init__(self, emb_size, conv_in_size, conv_out_size, output_size, kernel_size=3, stride=1, feature_crossed=False):
        """
        parameter example:
        emb_size = args.bert_feature_dim
        conv_in_size = args.bert_feature_dim//4
        conv_out_size = args.bert_feature_dim//8
        output_size = args.bert_feature_dim//8
        """
        super(CNN, self).__init__()
        self.feature_crossed = feature_crossed
        if feature_crossed:
            self.linear_transformer = nn.Linear(emb_size, conv_in_size)
        else:
            self.linear_transformer = nn.Linear(emb_size, conv_in_size//2)
        """
        infer padding
        """
        padding = (kernel_size - 1)//2
        self.conv2d = nn.Conv2d(conv_in_size, conv_out_size, kernel_size, stride, padding)
        self.final_transformer = nn.Linear(conv_out_size*2, output_size)
    
    def forward(self, feature):
        """
        params:
        feature: WHEN feature_crossed=False THEN (batch_size, num_sents, emb_size)
                 ELSE batch_size, num_sents, emb_size
        return:
        cross_feature: (batch_size, num_sents, num_sents, output_size)
        """
        if not self.feature_crossed:
            _, num_sents, _ = feature.size()
            feature = self.linear_transformer(feature)
            expanded_feature = feature.unsqueeze(2).expand([-1, -1, num_sents, -1])
            expanded_feature_t = feature.unsqueeze(1).expand([-1, num_sents, -1, -1])
            cross_feature = torch.cat((expanded_feature, expanded_feature_t), dim=3)
        else:
            _, num_sents, _, _ = feature.size()
            cross_feature = self.linear_transformer(feature)
        cnn_feature = cross_feature.permute(0, 3, 1, 2)
        cnn_out_feature = F.relu(self.conv2d(cnn_feature)).permute(0, 2, 3, 1)
        maxpooling_a, _ = torch.max(cnn_out_feature, dim=1)
        maxpooling_b, _ = torch.max(cnn_out_feature, dim=2)
        maxpooling_c = torch.cat([maxpooling_a.unsqueeze(3), maxpooling_b.unsqueeze(3)], dim=3)
        maxpool_feature, _ = torch.max(maxpooling_c, dim=3)
        expanded_maxpool_feature = maxpool_feature.unsqueeze(2).expand([-1, -1, num_sents, -1])
        expanded_maxpool_feature_t = maxpool_feature.unsqueeze(1).expand([-1, num_sents, -1, -1])
        cross_feature = torch.cat((expanded_maxpool_feature, expanded_maxpool_feature_t), dim=3)

        return self.final_transformer(cross_feature)