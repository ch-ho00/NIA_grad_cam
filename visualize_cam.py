import argparse
import yaml 
import os
import matplotlib.pyplot as plt

from torchvision.models import resnet50, resnet18
import torch

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from NIA_insta.core.dataset import Insta_Trend_Dataset
from NIA_insta.core.model import NIA_Wrapped, ConvGru, Wrapped_Model, custom_feature_extractor
from NIA_insta.core.function.train import get_img_sequence_tensor
from SingleImgNet import SImgNet

def load_checkpoint(cfg, model, checkpoint):
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        model.to(device)
        checkpoint = torch.load(checkpoint)
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except:
            try:
                model.module.load_state_dict(checkpoint['state_dict'])
            except:
                model.load_state_dict(checkpoint)
                return model
        cfg['train']['begin_epoch'] = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint)
        return model


def get_high_conf_idxs(cfg, model, valid_dataset, train_ref_dataset, thresh, device, show_prob_dist=False, hist_save_dir=None, prefix=None):
    softmax = torch.nn.Softmax(dim=1)
    tp_idxs = []
    tn_idxs = []
    fp_idxs = []
    fn_idxs = []
    tp_scores = [] 
    tn_scores = []
    fp_scores = []
    fn_scores = []

    ref_img_tensor = None
    correct = 0
    total = 0
    with torch.no_grad():
        for idx in range(len(valid_dataset)):
            input_, target = valid_dataset[idx]
            target = target.to(device)
            pair, ts, _ = input_
            if model.nia_model.concat_imgs: # if concat images then pair is a tensor and not list
                pair = pair.to(device)
            else:
                pair = [img.to(device) for img in pair]

            # img = img.cuda()
            if train_ref_dataset is not None:
                ref_idxs = train_ref_dataset.get_reference_set(time_interval=[ts - cfg['dataset']['reference_grace_period'], ts], 
                                                                return_type="sampled", 
                                                                sample_num=cfg['dataset']['reference_sample_num'])
                ref_img_tensor = get_img_sequence_tensor(train_ref_dataset, ref_idxs)
                if(ref_img_tensor is None):
                    continue
                ref_img_tensor = ref_img_tensor.to(device)
            ypred, trend_embed = model(pair, ref_img_tensor) # individual output shape: (1, 2)
            if ypred.shape[1] == 2:
                max_val, prediction = torch.max(softmax(ypred), 1)
                max_val = max_val.cpu()
            else:
                max_val = float(ypred.cpu())
                prediction = int(ypred.cpu() > 0)
                target = int(target.cpu() >0)

            if trend_embed is not None:
                trend_embed = trend_embed.cpu()
            #TODO replace with dictionary for later
            total+=1
            if prediction == 1:
                if target == 1:
                    correct += 1
                    if abs(max_val) > thresh:
                        tp_idxs.append([idx, float(max_val), trend_embed])
                    tp_scores.append(max_val)
                else:
                    if abs(max_val) > thresh:
                        fn_idxs.append([idx, float(max_val), trend_embed])
                    fn_scores.append(max_val)
            else:
                if target == 1:
                    if abs(max_val) > thresh:
                        fp_idxs.append([idx, float(max_val), trend_embed])
                    fp_scores.append(max_val)
                else:
                    correct+=1
                    if abs(max_val) > thresh:
                        tn_idxs.append([idx, float(max_val), trend_embed])
                    tn_scores.append(max_val)
        print(f'Validation Accuracy of Model: {correct}/{total} = {correct/total:.3f}')
        if show_prob_dist:
            prefix = prefix if prefix else ''
            hist_save_dir = os.path.join(hist_save_dir, 'hist')
            if not os.path.isdir(hist_save_dir):
                os.mkdir(hist_save_dir)

            plt.hist(tp_scores, bins=10)
            plt.savefig(os.path.join(hist_save_dir,f'{prefix}_tp_hist.png'))
            plt.clf()
            
            plt.hist(tn_scores, bins=10)
            plt.savefig(os.path.join(hist_save_dir,f'{prefix}_tn_hist.png'))
            plt.clf()
            
            plt.hist(fp_scores, bins=10)
            plt.savefig(os.path.join(hist_save_dir,f'{prefix}_fp_hist.png'))
            plt.clf()

            plt.hist(fn_scores, bins=10)
            plt.savefig(os.path.join(hist_save_dir,f'{prefix}_fn_hist.png'))
            plt.clf()

    return tp_idxs, tn_idxs, fp_idxs, fn_idxs

def save_activation(cam, list_, save_dir, account_id, type_, dataset): 
    '''
    Input
        cam : grad-cam model
        list_ : np.ndarray where each columns represent [idx, softmax score (i.e. probability), trend embedding]
        save_dir : directory to save the resultant activation imgs
        type_ : description of the indexes being sent through list_ (e.g. true-positive, false-negative, etc)
        dataset: the dataset to load the img tensor and target
    '''
    type2str = {
                'tp' : 'True Positive',
                'fp' : 'False Positive',
                'fn' : 'False Negative',
                'tn' : 'True Negative'
            }
    type2target = {
            'tp': ['pair', 'target'],
            'fp': ['target', 'pair'],
            'fn': ['pair', 'target'],
            'tn': ['target', 'pair'],
    }
    root = os.path.join(save_dir, type_)
    save_prefix = os.path.join(root, account_id)
    
    if not os.path.isdir(root):
        os.mkdir(root)
    if not os.path.isdir(save_prefix):
        os.mkdir(save_prefix)

    for idx, prob, trend_embed in list_:

        input_, target = dataset[idx]
        input_tensor, timestamp, unnorm_img = input_

        grayscale_cam = cam(input_tensor=input_tensor[0].detach().cpu().unsqueeze(0), target_category=int(target), embed=trend_embed)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(unnorm_img[0][0].permute(1,2,0).detach().cpu().numpy(), grayscale_cam, use_rgb=True)

        # save img
        fn = f'pair_{idx}_{unnorm_img[0][1][:-4]}.png'
        save_dir = os.path.join(save_prefix, fn)
        plt.imshow(visualization)
        plt.title(f'{type2str[type_]} {type2target[type_][0]}: Pred = {prob:.3f}, Target = {float(target) if target.dtype == torch.float32 else int(target):.3f}')
        plt.savefig(save_dir)          

        # switch to second img
        cam.model.switch_weight()
        grayscale_cam = cam(input_tensor=input_tensor[1].detach().cpu().unsqueeze(0), target_category=int(target), embed=trend_embed)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(unnorm_img[1][0].permute(1,2,0).detach().cpu().numpy(), grayscale_cam, use_rgb=True)
        # save img
        fn = f'pair_{idx}_{unnorm_img[1][1][:-4]}.png'
        save_dir = os.path.join(save_prefix, fn)
        plt.imshow(visualization)
        plt.title(f'{type2str[type_]} {type2target[type_][1]}: Pred = {prob:.3f}, Target = {float(target) if target.dtype == torch.float32 else int(target):.3f}')
        plt.savefig(save_dir)          
        cam.model.switch_weight()


def main(args):
    print(f"=============================\n Visualization of {args.prefix} \n===========================")
    cfg = yaml.safe_load(open(args.cfg, 'r'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model
    if cfg['model']['name'] == 'resnet50':
        nia_model = models.resnet50(pretrained=cfg['model']['pretrained'])
        nia_model = NIA_Wrapped(nia_model, cfg['model']['type'], embed_size=cfg['embed_model']['final_embedding_size'], concat_imgs=cfg['dataset']['args']['concat_imgs'])

    elif cfg['model']['name'] == 'resnet18':
        nia_model = models.resnet18(pretrained=cfg['model']['pretrained'])
        nia_model = NIA_Wrapped(nia_model, cfg['model']['type'], embed_size=cfg['embed_model']['final_embedding_size'], concat_imgs=cfg['dataset']['args']['concat_imgs'])
    elif cfg['model']['name'] == 'custom':
        nia_model = custom_feature_extractor()
        nia_model = NIA_Wrapped(nia_model, cfg['model']['type'], 
                                embed_size=cfg['embed_model']['final_embedding_size'] if cfg['use_img_embedding'] else 0, 
                                concat_imgs=cfg['dataset']['args']['concat_imgs'])
    else:
        # TODO: add new model
        pass

    embed_model = None
    if(cfg['use_img_embedding']):
        if cfg['embed_model']['name'] == 'ConvGru':
            del cfg['embed_model']['name']
            embed_model = ConvGru(**cfg['embed_model'])
        else:
            raise NotImplementedError
    model = Wrapped_Model(nia_model, embed_model)
    model = load_checkpoint(cfg, model, args.checkpoint)
    model = model.to(device)
    
    #====#
    if(model.trend_embed_model is not None):
        print("Embedding model attached")
    else:
        print("No embedding model found")
    #====#
    
    #=========================#
    # Loading dataset 
    train_dataset, valid_dataset = eval(cfg['dataset']['dataset']).get_data_loaders(cfg, return_dataset=True)
    train_ref_dataset = None
    val_ref_dataset   = None
    if(cfg['use_img_embedding']):
        train_ref_dataset, val_ref_dataset = eval(cfg['dataset']['dataset']).get_data_loaders(cfg, reference_dataset=True)
    
    #=========================#
 
    # idxs with prediction probability higher than threshold
    idxs = get_high_conf_idxs(cfg, model, valid_dataset, train_ref_dataset, args.threshold, device, True, args.savedir, args.prefix)
    tp_idxs, tn_idxs, fp_idxs, fn_idxs = idxs
    
    print(f"Number of TP/TN/FP/FN : {len(tp_idxs)}/{len(tn_idxs)}/{len(fp_idxs)}/{len(fn_idxs)}")    
    single_img_model = SImgNet(model, cfg['model']['type'])
    if cfg['model']['name'] == 'custom':
        target_layer = single_img_model.backbone.middle_model[-4]
    elif 'resnet' in cfg['model']['name']:
        target_layer = single_img_model.backbone.layer4[-1]

    cpu_device = torch.device('cpu')
    single_img_model.to(cpu_device)
    cam = GradCAM(model=single_img_model, target_layer=target_layer)

    # save activation
    save_activation(cam, tp_idxs, args.savedir, args.prefix, 'tp', valid_dataset)
    save_activation(cam, tn_idxs, args.savedir, args.prefix, 'tn', valid_dataset)
    save_activation(cam, fp_idxs, args.savedir, args.prefix, 'fp', valid_dataset)
    save_activation(cam, fn_idxs, args.savedir, args.prefix, 'fn', valid_dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # general
    parser.add_argument('--cfg',
                        help='Dataset, Model Configuration',
                        required=True,
                        type=str)
    parser.add_argument('--checkpoint',
                        help='Directory to the checkpoint pth',
                        required=True,
                        type=str)
    parser.add_argument('--threshold',
                        help='Threshold of prediction probability for visualization',
                        required=True,
                        type=float)
    parser.add_argument('--savedir',
                    help='Directory to save the heatmap visualization',
                    required=True,
                    type=str)
    parser.add_argument('--prefix',
                    help='Name of subfolder to be stored',
                    required=True,
                    type=str)

    args = parser.parse_args()
    main(args)