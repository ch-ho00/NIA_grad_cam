import torch
import torch.nn as nn

class SImgNet(nn.Module):
    def __init__(self,orig_model, type_, first_img=True):
        super(SImgNet, self).__init__()
        self.orig_model = orig_model

        self.backbone = orig_model.nia_model.feat_extractor
        img_embed_dim = orig_model.nia_model.img_embed_dim

        self.fc = nn.Linear(img_embed_dim + orig_model.nia_model.embed_size, 2 if type_=='classification' else 1)
        self.first_img = first_img
        self.img_embed_dim = img_embed_dim
        with torch.no_grad():
            if first_img:
                self.fc.weight[:, :img_embed_dim] = orig_model.nia_model.fc.weight.detach()[:, :img_embed_dim]
                self.fc.weight[:, img_embed_dim:] = orig_model.nia_model.fc.weight.detach()[:, img_embed_dim*2:]
            else:
                self.fc.weight = orig_model.nia_model.fc.weight.detach()[:, img_embed_dim:]

    def switch_weight(self):
        with torch.no_grad():        
            if self.first_img:
                self.fc.weight = nn.Parameter(self.orig_model.nia_model.fc.weight.detach()[:, self.img_embed_dim:])
            else:
                self.fc.weight[:, :self.img_embed_dim] = nn.Parameter(self.orig_model.nia_model.fc.weight.detach()[:, :self.img_embed_dim])
                self.fc.weight[:, self.img_embed_dim:] = nn.Parameter(self.orig_model.nia_model.fc.weight.detach()[:, self.img_embed_dim*2:])
        self.first_img = not self.first_img

    def __call__(self, x, trend_embed=None):
        img_feat = self.backbone(x)
        if trend_embed is not None:
            img_feat = torch.cat([img_feat, trend_embed], axis=1)

        return self.fc(img_feat)
