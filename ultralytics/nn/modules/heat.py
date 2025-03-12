import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def make_divisible(x, divisor) -> int:
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


class Segmenter(nn.Module):
    def __init__(self, nc=1, ch=128):
        super(Segmenter, self).__init__()
        self.m = nn.Conv2d(ch, nc, 1)  # output conv

    def forward(self, x):
        return self.m(x)



class HeatMapParser(nn.Module):
    def __init__(self, c1,c2, ratio=10, threshold=0.5, mask_only=True, cluster_only=False):
        super().__init__()
        self.c = c2
        self.ratio = ratio
        self.threshold = threshold
        self.mask_only = mask_only
        self.cluster_only = cluster_only
        self.segm =  Segmenter(1,c2)

        self.grid = None

    def forward(self, x):
        x, heatmaps = x
        heatmaps = self.segm(heatmaps)
        bs, c, ny, nx = x.shape
        device, dtype = x.device, x.dtype
        assert c == self.c, f'{c} - {self.c}'
        # assert len(heatmaps) <= 3
        # if heatmaps.ndimension() == 3:
        #     mask_pred = heatmaps[0].detach().unsqueeze(1)
        # else:
        mask_pred = heatmaps[0].detach()

        if torch.max(mask_pred) > 1. or torch.min(mask_pred) < 0.:
            mask_pred = mask_pred.sigmoid()
        #
        # mask_pred = mask_pred[:, 0, :, :].detach()
        try:
            mask_pred = mask_pred[:, 0, :, :].detach()
        except Exception as e:
            mask_pred = mask_pred.unsqueeze(1)[:, 0, :, :].detach()
        # if getattr(self, 'mask_only', False):
        #     return x

        # t0 = time_synchronized()
        if self.training:
            # if getattr(self, 'cluster_only', False):
            #     total_clusters = self.ada_slicer(mask_pred, self.ratio, self.threshold * 1. + 0.)
            #     return self.get_offsets_by_clusters(total_clusters).to(device)
            output = self.uni_slicer(x, mask_pred, self.ratio, self.threshold * 1. + 0., device=device)
            return output[0]
        else:
            # total_clusters = self.ada_slicer(mask_pred, self.ratio, self.threshold * 1.0 + 0.)
            total_clusters = self.ada_slicer_fast(mask_pred, self.ratio, self.threshold * 1.0 + 0.)
            # if getattr(self, 'cluster_only', False):
            #     return self.get_offsets_by_clusters(total_clusters).to(device)
            # t1 = time_synchronized()
            patches, offsets = [], []
            for bi, clusters in enumerate(total_clusters):
                for x1, y1, x2, y2 in clusters:
                    patches.append(x[bi, :, y1:y2, x1:x2])
                    offsets.append(torch.tensor([bi, x1, y1, x2, y2]))
                    assert patches[-1].shape[-2] == y2 - y1 and patches[-1].shape[-1] == x2 - x1, '%f %f %f %f' % (
                    x1, y1, x2, y2)
            # t2 = time_synchronized()
            # print(f'Patchify: {t1 - t0:.3f}s. Slice:{t2 - t1:.3f}s')

            if len(patches):
                return torch.stack(patches), torch.stack(offsets).to(device)
            else:
                return torch.zeros((0, c, ny, nx), device=device), torch.zeros((0, 5), device=device)

    @staticmethod
    def get_offsets_by_clusters(total_clusters):
        offsets = []
        for bi, clusters in enumerate(total_clusters):
            b = torch.full_like(clusters[:, :1], bi)
            offsets.append(torch.cat((b, clusters), dim=1))
        return torch.cat(offsets)

    @torch.no_grad()
    def ada_slicer(self, mask_pred: torch.Tensor, ratio=8, threshold=0.3):  # better
        # t0 = time_synchronized()
        bs, height, width = mask_pred.shape
        device, dtype = mask_pred.device, mask_pred.dtype
        cluster_wh = max(make_divisible(width / ratio, 4), make_divisible(height / ratio, 4))  # 保证正方形
        cluster_w, cluster_h = cluster_wh, cluster_wh
        # cluster_w, cluster_h = make_divisible(width / ratio, 4), make_divisible(height / ratio, 4)
        half_clus_w, half_clus_h = cluster_w // 2, cluster_h // 2
        outs = []

        # t1 = time_synchronized()
        activated = mask_pred >= threshold
        maxima = F.max_pool2d(mask_pred, 3, stride=1, padding=1) == mask_pred
        obj_centers = activated & maxima
        padding = half_clus_w // 2
        obj_sizes = F.avg_pool2d(mask_pred, padding * 2 + 1, stride=1, padding=padding)

        # bi, yi, xi
        # t2 = time_synchronized()
        cb, cy, cx = obj_centers.nonzero(as_tuple=True)
        obj_sizes = obj_sizes[cb, cy, cx]

        # t3 = time_synchronized()
        for bi in range(bs):
            ci = cb == bi
            cn = ci.sum().item()
            if cn == 0:
                outs.append(torch.zeros((0, 4), device=device))
                continue

            if bs == 1:
                sizes = obj_sizes
                cy_bi, cx_bi = cy, cx
            else:
                sizes = obj_sizes[ci]
                cy_bi, cx_bi = cy[ci], cx[ci]

            # shape(n,1)
            init_x1 = cx_bi.clamp(half_clus_w, width - half_clus_w) - half_clus_w
            init_y1 = cy_bi.clamp(half_clus_h, height - half_clus_h) - half_clus_h

            # shape(1,m)
            if not hasattr(self, 'grid') or self.grid is None or self.grid[0].shape[-1] != cluster_h * cluster_w:
                gy, gx = torch.meshgrid(torch.arange(cluster_h), torch.arange(cluster_w))
                self.grid = (gy.reshape(1, -1).to(device), gx.reshape(1, -1).to(device))
            gy, gx = self.grid

            # shape(n,m)
            act_x, act_y = (init_x1.view(-1, 1) + gx).view(-1), (init_y1.view(-1, 1) + gy).view(-1)
            act = activated[bi, act_y, act_x].view(cn, cluster_h, cluster_w)

            # t4 = time_synchronized()
            act_x, act_y = act.any(dim=1).long(), act.any(dim=2).long()
            dx1, dx2 = (1 - act_x).argmin(dim=1), -(1 - act_x.flip((1,))).argmin(dim=1)
            dy1, dy2 = (1 - act_y).argmin(dim=1), -(1 - act_y.flip((1,))).argmin(dim=1)
            dx = torch.where(dx1.abs() > dx2.abs(), dx1, dx2)
            dy = torch.where(dy1.abs() > dy2.abs(), dy1, dy2)

            # t5 = time_synchronized()
            refine_x1, refine_y1 = (init_x1 + dx).clamp(0, width - cluster_w).to(dtype), \
                (init_y1 + dy).clamp(0, height - cluster_h).to(dtype)
            refine_x2, refine_y2 = refine_x1 + cluster_w, refine_y1 + cluster_h
            total_clusters = torch.stack((refine_x1, refine_y1, refine_x2, refine_y2), dim=1).long()

            # i = torchvision.ops.nms(total_clusters, sizes, 0.8)  # NMS
            # clusters = total_clusters[i].long()

            # t6 = time_synchronized()
            overlap = (refine_x1[:, None] <= cx_bi[None, :]) & (cx_bi[None, :] < refine_x2[:, None]) & \
                      (refine_y1[:, None] <= cy_bi[None, :]) & (cy_bi[None, :] < refine_y2[:, None])
            clusters = []
            contained = torch.full_like(overlap[0], False)
            for max_i in torch.argsort(sizes, descending=True):
                if contained[max_i]:
                    continue
                clusters.append(total_clusters[max_i])
                contained |= overlap[max_i]

            # t7 = time_synchronized()
            outs.append(torch.stack(clusters) if len(clusters) else torch.zeros_like(total_clusters[:0, :]))

            # print(f't1: {(t1-t0)*1000:.3f}, t2: {(t2-t1)*1000:.3f}, t3: {(t3-t2)*1000:.3f}, t4: {(t4-t3)*1000:.3f}, t5: {(t5-t4)*1000:.3f}, t6: {(t6-t5)*1000:.3f}, t7: {(t7-t6)*1000:.3f}')
        return outs

    @torch.no_grad()
    def ada_slicer_fast(self, mask_pred: torch.Tensor, ratio=8, threshold=0.3):  # faster
        # t0 = time_synchronized()
        bs, height, width = mask_pred.shape
        # assert width % ratio == 0 and height % ratio == 0, f'{width} // {height}'
        device, dtype = mask_pred.device, mask_pred.dtype
        # cluster_wh = max(make_divisible(width / ratio, 4), make_divisible(height / ratio, 4))  # 保证正方形
        # cluster_w, cluster_h = cluster_wh, cluster_wh
        cluster_w, cluster_h = make_divisible(width / ratio, 4), make_divisible(height / ratio, 4)
        # cluster_w, cluster_h = width // ratio, height // ratio
        # assert cluster_w % 4 == 0 and cluster_h % 4 == 0, f'{width} -> {cluster_w} // {height} -> {cluster_h}'
        ratio_x, ratio_y = int(math.ceil(width / cluster_w)), int(math.ceil(height / cluster_h))
        half_clus_w, half_clus_h = cluster_w // 2, cluster_h // 2
        outs = []

        if getattr(self, 'grid_vtx', None) is None or self.grid_vtx.size(0) != ratio_x * ratio_y * bs:
            gy, gx = torch.meshgrid(torch.arange(ratio_y), torch.arange(ratio_x))
            gxy = torch.stack((gy.reshape(-1), gx.reshape(-1)), dim=1).unsqueeze(0).repeat(bs, 1, 1).view(-1,
                                                                                                          2)  # shape(bs*8*8,2)
            gb = torch.arange(bs).view(-1, 1).repeat(1, ratio_x * ratio_y).view(-1, 1)  # shape(bs*8*8, 1)
            self.grid_vtx = torch.cat((gb, gxy), dim=1).to(device)  # shape(bs*8*8, 3)
        rb, ry, rx = self.grid_vtx.T

        if getattr(self, 'grid', None) is None or self.grid[0].shape[-1] != cluster_h * cluster_w:
            gy, gx = torch.meshgrid(torch.arange(cluster_h), torch.arange(cluster_w))
            self.grid = (gy.reshape(1, -1).to(device), gx.reshape(1, -1).to(device))
        gy, gx = self.grid

        # t1 = time_synchronized()
        activated = mask_pred >= threshold
        maxima: torch.Tensor = F.max_pool2d(mask_pred, 3, stride=1, padding=1) == mask_pred
        obj_centers = activated & maxima
        if (~obj_centers).all():
            return [torch.zeros((0, 4), device=device) for _ in range(bs)]
        padding = max(half_clus_w, half_clus_h) // 2
        obj_sizes = F.avg_pool2d(mask_pred, padding * 2 + 1, stride=1, padding=padding)

        valid_regions = F.pad(obj_centers, (0, ratio_x * cluster_w - width, 0, ratio_y * cluster_h - height))
        valid_regions = F.max_pool2d(valid_regions.float(), (cluster_h, cluster_w), stride=(cluster_h, cluster_w),
                                     padding=0)
        valid_regions = valid_regions.view(-1) > 0
        cb, x1, y1 = rb[valid_regions], rx[valid_regions] * cluster_w, ry[valid_regions] * cluster_h

        act_x, act_y = (x1.view(-1, 1) + gx).view(-1), (y1.view(-1, 1) + gy).view(-1)
        act_b = cb.view(-1, 1).repeat((1, gy.size(1))).view(-1)
        activated = F.pad(activated, (0, ratio_x * cluster_w - width, 0, ratio_y * cluster_h - height))
        act = activated[act_b, act_y, act_x].view(cb.shape[0], cluster_h, cluster_w)

        act_x, act_y = act.any(dim=1).long(), act.any(dim=2).long()  # shape(nc, cw), shape(nc, ch)
        dx1, dx2 = (1 - act_x).argmin(dim=1), -(1 - act_x.flip((1,))).argmin(dim=1)
        dy1, dy2 = (1 - act_y).argmin(dim=1), -(1 - act_y.flip((1,))).argmin(dim=1)
        dx = torch.where(dx1.abs() > dx2.abs(), dx1, dx2)
        dy = torch.where(dy1.abs() > dy2.abs(), dy1, dy2)

        # t5 = time_synchronized()
        x1, y1 = (x1 + dx).clamp(0, width - cluster_w), \
            (y1 + dy).clamp(0, height - cluster_h)
        x2, y2 = x1 + cluster_w, y1 + cluster_h
        bboxes = torch.stack((x1, y1, x2, y2), dim=1).long()

        # offsets = (cb * max(width, height)).unsqueeze(1)
        # scores = obj_sizes[cb, y1 + half_clus_h, x1 + half_clus_w]
        # indices = torchvision.ops.nms((bboxes + offsets).float(), scores, iou_threshold=0.9)  # 0.65
        # cb, bboxes = cb[indices], bboxes[indices]

        for bi in range(bs):
            outs.append(bboxes[cb == bi])

        return outs

    def uni_slicer(self, feat, mask_pred, ratio=10, threshold=0.3, device='cuda'):
        def _slice(x: torch.Tensor):
            # if len(x.shape) == 4:
            #     b, c, h, w = x.shape
            #     return x.view(b, c, ratio, h//ratio, ratio, w//ratio).permute(0,2,4,1,3,5).contiguous().view(b*ratio*ratio, c, h//ratio, w//ratio)
            # else:
            #     b, h, w = x.shape
            #     return x.view(b, ratio, h//ratio, ratio, w//ratio).transpose(2,3).contiguous().view(b*ratio*ratio, h//ratio, w//ratio)

            x_list = torch.chunk(x, ratio, dim=-2)  # [shape(bs,c,h//8,w)] * 8
            y = []
            for x in x_list:
                y.extend(torch.chunk(x, ratio, dim=-1))  # [shape(bs,c,h//8,w//8)] * 8
            return torch.cat(y, dim=0)  # shape(8*8*bs,c,h//8,w//8)

        bs, height, width = mask_pred.shape
        assert height == width
        assert width % (ratio * 4) == 0 and height % (ratio * 4) == 0, f'{width}, {height}'
        cluster_wh = max(make_divisible(width / ratio, 4), make_divisible(height / ratio, 4))  # 保证正方形

        if not hasattr(self, 'grid_off') or len(self.grid_off) != bs * ratio * ratio or self.grid_off.device != device:
            xrange = torch.arange(ratio)
            gy, gx = torch.meshgrid(xrange, xrange)
            gxy = torch.stack((gy.reshape(-1), gx.reshape(-1)), dim=1).unsqueeze(1).repeat(1, bs, 1).view(-1,
                                                                                                          2)  # shape(8*8*bs,2)
            gb = torch.arange(bs).view(1, -1).repeat(ratio ** 2, 1).view(-1, 1)  # shape(8*8*bs)
            gy, gx = gxy.T
            grid = torch.stack((gx, gy, gx + 1, gy + 1), dim=-1) * cluster_wh
            self.grid_off = torch.cat((gb, grid), dim=1).to(device)

        if getattr(self, 'cluster_only', False):
            return self.grid_off

        patches = _slice(feat)  # shape(8*8*bs,c,h//8,w//8)

        return patches, self.grid_off

        # activated = mask_pred >= threshold
        # maxima = F.max_pool2d(mask_pred, 3, stride=1, padding=1) == mask_pred
        # obj_centers = activated & maxima

        # mask = _slice(obj_centers)  # shape(8*8*bs,h//8,w//8)
        # indices = mask.view(len(patches), -1).any(dim=1)
        # return patches[indices], self.grid_off[indices]