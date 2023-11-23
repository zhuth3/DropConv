import torch
import torch.nn as nn
import spconv.pytorch as spconv
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from pcdet.models.backbones_3d.drop_sparse_conv.utils import split_voxels_sparse
from pcdet.utils import common_utils
import math
from .utils import FocalLoss

class DropSubMConv(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, voxel_stride, stride=1, norm_fn=None, indice_key=None,
                image_channel=3, kernel_size=3, padding=1, mask_multi=False, use_img=False,
                topk=False, threshold=0.5, enlarge_voxel_channels=-1, 
                point_cloud_range=[-3, -40, 0, 1, 40, 70.4],
                voxel_size = [0.1, 0.05, 0.05], is_first_layer=False, is_last_layer=False):
        super(DropSubMConv, self).__init__()

        self.voxel_size = torch.Tensor(voxel_size).cuda()
        self.inv_idx =  torch.Tensor([2, 1, 0]).long().cuda()
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()

        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU(True)
        offset_channels = 1

        self.topk = topk
        self.threshold = threshold
        self.voxel_stride = voxel_stride
        self.focal_loss = FocalLoss()
        self.mask_multi = mask_multi
        self.use_img = use_img
        self.stride = stride
        self.padding = padding
        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer

        voxel_channel = enlarge_voxel_channels if enlarge_voxel_channels>0 else inplanes
        in_channels = image_channel + voxel_channel if use_img else voxel_channel

        self.conv_enlarge = spconv.SparseSequential(spconv.SubMConv3d(inplanes, enlarge_voxel_channels, 
            kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key+'_enlarge'),
            norm_fn(enlarge_voxel_channels),
            nn.ReLU(True)) if enlarge_voxel_channels>0 else None

        self.conv_imp = spconv.SubMConv3d(in_channels, planes+1, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key+'_imp')
        
    def _gen_sparse_features(self, x, batch_dict, voxels_3d):
        """
            Generate the output sparse features from the focal sparse conv.
            Args:
                x: [N, C], lidar sparse features
                batch_dict: input and output information during forward
                voxels_3d: [N, 3], the 3d positions of voxel centers
        """
        batch_size = x.batch_size
        voxel_features_fore = []
        voxel_indices_fore = []
        voxel_features_back = []
        voxel_indices_back = []
        voxel_coords_fore = []
        mask_voxels = []
        box_of_pts_cls_targets = []
        
        loss_box_of_pts = 0
        for b in range(batch_size):
            index = x.indices[:, 0]
            batch_index = index==b
            indices_ori = x.indices[batch_index]
            features_ori = x.features[batch_index, :]

            features_fore, indices_fore, features_back, indices_back, mask_voxel = split_voxels_sparse(indices_ori, features_ori, mask_multi=self.mask_multi, topk=self.topk, threshold=self.threshold)

            if self.training:
                voxels_3d_batch = voxels_3d[batch_index].unsqueeze(0)
                gt_boxes = batch_dict['gt_boxes'][b, :, :7].unsqueeze(0)
                box_of_pts_batch = points_in_boxes_gpu(voxels_3d_batch[:, :, self.inv_idx], gt_boxes).squeeze(0)
                idx = box_of_pts_batch>=0
                box_of_pts_batch[idx] = 1
                idx = box_of_pts_batch<0
                box_of_pts_batch[idx] = 0
                box_of_pts_cls_targets.append(box_of_pts_batch)

            voxel_features_fore.append(features_fore)
            voxel_indices_fore.append(indices_fore)
            voxel_features_back.append(features_back)
            voxel_indices_back.append(indices_back)
            mask_voxels.append(mask_voxel)

        voxel_features_fore = torch.cat(voxel_features_fore, dim=0)
        voxel_indices_fore = torch.cat(voxel_indices_fore, dim=0)
        voxel_features_back = torch.cat(voxel_features_back, dim=0)
        voxel_indices_back = torch.cat(voxel_indices_back, dim=0)

        x_fore = spconv.SparseConvTensor(voxel_features_fore, voxel_indices_fore, x.spatial_shape, x.batch_size)
        x_back = spconv.SparseConvTensor(voxel_features_back, voxel_indices_back, x.spatial_shape, x.batch_size)

        loss_box_of_pts = 0
        if self.training:
            mask_voxels = torch.cat(mask_voxels)
            box_of_pts_cls_targets = torch.cat(box_of_pts_cls_targets)
            mask_voxels_two_classes = torch.cat([1-mask_voxels.unsqueeze(-1), mask_voxels.unsqueeze(-1)], dim=1)
            loss_box_of_pts = self.focal_loss(mask_voxels_two_classes, box_of_pts_cls_targets.long())

        return x_fore, x_back, batch_dict, loss_box_of_pts
  
    def forward(self, x, batch_dict, x_rgb=None):
        spatial_indices = x.indices[:, 1:] * self.voxel_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]

        x_predict = self.conv_enlarge(x) if self.conv_enlarge else x
        imps_3d = self.conv_imp(x_predict)

        x_fore, x_back, batch_dict, loss_box_of_pts = self._gen_sparse_features(imps_3d, batch_dict, voxels_3d)

        out = x_fore
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        return out, batch_dict, loss_box_of_pts

