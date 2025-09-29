import os,sys,io
import numpy as np
from copy import deepcopy
import torch
from tqdm import tqdm
import pickle
import re

########################################################################################################################

def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =',end=' ')
    count=0
    for p in model.parameters():
        print(p.size(),end=' ')
        count+=np.prod(p.size())
    print()
    print('Num parameters = %s'%(human_format(count)))
    print('-'*100)
    return count

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim,'=',end=' ')
        opt=optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n+':',opt[n],end=', ')
        print()
    return

########################################################################################################################

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

########################################################################################################################

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

########################################################################################################################

def compute_mean_std_dataset(dataset):
    # dataset already put ToTensor
    mean=0
    std=0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for image, _ in loader:
        mean+=image.mean(3).mean(2)
    mean /= len(dataset)

    mean_expanded=mean.view(mean.size(0),mean.size(1),1,1).expand_as(image)
    for image, _ in loader:
        std+=(image-mean_expanded).pow(2).sum(3).sum(2)

    std=(std/(len(dataset)*image.size(2)*image.size(3)-1)).sqrt()

    return mean, std

########################################################################################################################

def fisher_matrix_diag(t,x,y,model,criterion,sbatch=20):
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()
    for i in tqdm(range(0,x.size(0),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()
        images=torch.autograd.Variable(x[b],volatile=False)
        target=torch.autograd.Variable(y[b],volatile=False)
        # Forward and backward
        model.zero_grad()
        outputs=model.forward(images)
        loss=criterion(t,outputs[t],target)
        loss.backward()
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    for n,_ in model.named_parameters():
        fisher[n]=fisher[n]/x.size(0)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    return fisher

########################################################################################################################

def cross_entropy(outputs,targets,exp=1,size_average=True,eps=1e-5):
    out=torch.nn.functional.softmax(outputs)
    tar=torch.nn.functional.softmax(targets)
    if exp!=1:
        out=out.pow(exp)
        out=out/out.sum(1).view(-1,1).expand_as(out)
        tar=tar.pow(exp)
        tar=tar/tar.sum(1).view(-1,1).expand_as(tar)
    out=out+eps/out.size(1)
    out=out/out.sum(1).view(-1,1).expand_as(out)
    ce=-(tar*out.log()).sum(1)
    if size_average:
        ce=ce.mean()
    return ce

########################################################################################################################

def set_req_grad(layer,req_grad):
    if hasattr(layer,'weight'):
        layer.weight.requires_grad=req_grad
    if hasattr(layer,'bias'):
        layer.bias.requires_grad=req_grad
    return

########################################################################################################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
########################################################################################################################

def fisher_matrix_diag_bert(t,train,device,model,criterion,sbatch=20,scenario='til',imp='loss',adjust_final=False,imp_layer_norm=False,get_grad_dir=False):
    # Init
    fisher={}
    grad_dir={}
    fisher_true={}
    train_true=0
    train_false=0
    fisher_fal={}
    for n,p in model.named_parameters():
        # print(n)
        fisher[n]=0*p.data
        grad_dir[n]=0*p.data
        fisher_true[n]=0*p.data
        fisher_fal[n]=0*p.data
        
    # Compute
    model.train()

    for i in tqdm(range(0,len(train),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,len(train)])))#.cuda()
        batch=train[b]
        batch = [
            bat.to(device) if bat is not None else None for bat in batch]
        input_ids, segment_ids, input_mask, targets,_= batch

        # Forward and backward
        model.zero_grad()
        output_dict=model.forward(input_ids, segment_ids, input_mask)
        if 'til' in scenario:
            outputs=output_dict['y']
            output = outputs[t]
        elif 'cil' in scenario:
            output=output_dict['y']
        elif 'dil' in scenario:
            output=output_dict['y']
        
        _,pred=output.max(1)
        true_pred_idx=(pred==targets).nonzero().squeeze()
        false_pred_idx=(pred!=targets).nonzero().squeeze()
        train_true+=sum(pred==targets)
        train_false+=sum(pred!=targets)

        if imp=='loss':
            if adjust_final:
                if sum(pred==targets)>0:
                    loss=criterion(t,torch.index_select(output, dim=0, index=true_pred_idx),torch.index_select(targets, dim=0, index=true_pred_idx))
                    loss.backward(retain_graph=True)
                    # Get gradients
                    for n,p in model.named_parameters():
                        if p.grad is not None and 'last' in n: #Replaced 'layer.11' with 'last'
                            fisher_true[n]+=sbatch*p.grad.data.pow(2)
                    model.zero_grad()
                if sum(pred!=targets)>0:
                    loss=criterion(t,torch.index_select(output, dim=0, index=false_pred_idx),torch.index_select(targets, dim=0, index=false_pred_idx))
                    loss.backward(retain_graph=True)
                    # Get gradients
                    for n,p in model.named_parameters():
                        if p.grad is not None and 'last' in n:
                            fisher_fal[n]+=sbatch*p.grad.data.pow(2)
                    model.zero_grad()
                loss=criterion(t,output,targets)
                loss.backward # Think if this will be affected by the retain_graph=True
                # Get gradients
                for n,p in model.named_parameters():
                    if p.grad is not None:
                        if 'last' in n and sum(pred==targets)==0:
                            fisher_true[n]+=0
                        elif 'last' in n and sum(pred!=targets)==0:
                            fisher_fal[n]+=0
                        else:
                            fisher[n]+=sbatch*p.grad.data.pow(2)
            else:
                loss=criterion(t,output,targets)
                loss.backward()
                # Get gradients
                for n,p in model.named_parameters():
                    if p.grad is not None:
                        fisher[n]+=sbatch*p.grad.data.pow(2)
            
        elif imp=='function':
            if adjust_final:
                if sum(pred==targets)>0:
                    output1 = torch.index_select(output, dim=0, index=true_pred_idx).pow(2).sum(dim=1).sum()
                    output1.backward(retain_graph=True)
                    # Get gradients
                    for n,p in model.named_parameters():
                        if p.grad is not None and 'last' in n:
                            fisher_true[n]+=sbatch*torch.abs(p.grad.data)
                    model.zero_grad()
                if sum(pred!=targets)>0:
                    output2 = torch.index_select(output, dim=0, index=false_pred_idx).pow(2).sum(dim=1).sum()
                    output2.backward(retain_graph=True) # Think if this will be affected by the retain_graph=True
                    # Get gradients
                    for n,p in model.named_parameters():
                        if p.grad is not None and 'last' in n:
                            fisher_fal[n]+=sbatch*torch.abs(p.grad.data)
                    model.zero_grad()
                output = output.pow(2).sum(dim=1).sum()
                output.backward() # Think if this will be affected by the retain_graph=True
                # Get gradients
                for n,p in model.named_parameters():
                    if p.grad is not None:
                        if 'last' in n and sum(pred==targets)==0:
                            fisher_true[n]+=0
                        elif 'last' in n and sum(pred!=targets)==0:
                            fisher_fal[n]+=0
                        else:
                            fisher[n]+=sbatch*torch.abs(p.grad.data)
            else:
                # Square of the l2-norm: output.pow(2).sum(dim=1)
                # Calculate square of the l2-norm and then sum for all samples in the batch
                output = output.pow(2).sum(dim=1).sum()
                output.backward()
                # Get gradients
                for n,p in model.named_parameters():
                    if p.grad is not None:
                        fisher[n]+=sbatch*torch.abs(p.grad.data)
                        grad_dir[n]+=sbatch*p.grad.data
        
    # Mean
    for n,_ in model.named_parameters():
        if adjust_final and 'last' in n:
            # fisher[n]=fisher_true[n]/train_true #v18
            fisher_true[n] = fisher_true[n]/train_true
            fisher_fal[n] = fisher_fal[n]/train_false
            fisher_check = fisher_true[n]/(fisher_true[n]+fisher_fal[n])
            fisher[n][fisher_check>0.5] = fisher_true[n][fisher_check>0.5]
            fisher[n][fisher_check<=0.5] = 0.0000000000001 # Small value ~ 0
        else:
            fisher[n]=fisher[n]/len(train)
            fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
            grad_dir[n]=grad_dir[n]/len(train)
            # grad_dir[n]=torch.autograd.Variable(grad_dir[n],requires_grad=False)
            # if 'output.adapter' in n or 'output.LayerNorm' in n:
                # print(fisher[n])
    
    # Normalize by layer
    if imp_layer_norm:
        layer_range = {}
        layer_min = {}
        for i in range(12):
            wgts = torch.cat([
                fisher['bert.encoder.layer.'+str(i)+'.attention.output.LayerNorm.weight'].flatten()
                ,fisher['bert.encoder.layer.'+str(i)+'.attention.output.LayerNorm.bias'].flatten()
                ,fisher['bert.encoder.layer.'+str(i)+'.output.LayerNorm.weight'].flatten()
                ,fisher['bert.encoder.layer.'+str(i)+'.output.LayerNorm.bias'].flatten()
                ,fisher['bert.encoder.layer.'+str(i)+'.attention.output.adapter.fc1.weight'].flatten()
                ,fisher['bert.encoder.layer.'+str(i)+'.attention.output.adapter.fc1.bias'].flatten()
                ,fisher['bert.encoder.layer.'+str(i)+'.attention.output.adapter.fc2.weight'].flatten()
                ,fisher['bert.encoder.layer.'+str(i)+'.attention.output.adapter.fc2.bias'].flatten()
                ,fisher['bert.encoder.layer.'+str(i)+'.output.adapter.fc1.weight'].flatten()
                ,fisher['bert.encoder.layer.'+str(i)+'.output.adapter.fc1.bias'].flatten()
                ,fisher['bert.encoder.layer.'+str(i)+'.output.adapter.fc2.weight'].flatten()
                ,fisher['bert.encoder.layer.'+str(i)+'.output.adapter.fc2.bias'].flatten()
            ])
            # wgts=torch.hstack(wgts).flatten()
            assert len(wgts.shape)==1 # check that it is flattened
            layer_min[str(i)] = torch.min(wgts)
            layer_range[str(i)] = torch.max(wgts)-torch.min(wgts)
        
        for n,_ in model.named_parameters():
            if 'output.adapter' in n or 'output.LayerNorm' in n:
                i = re.findall("layer\.(\d+)\.",n)[0]
                fisher[n]=(fisher[n]-layer_min[i])/layer_range[i]
    if get_grad_dir:
        return fisher,grad_dir
    else:
        return fisher

########################################################################################################################################
# TODO: make this function dynamic?
def get_my_lambda(idrandom,t,class_counts):
    seen_lambda=1
    if idrandom==0:
        classes_seen = t*5
        classes_cur = 5
        classes_later = 30-(classes_seen+classes_cur)
        my_lambda = torch.cat([torch.ones(classes_seen)*seen_lambda,torch.ones(classes_cur)*torch.tensor(class_counts),torch.zeros(classes_later)], dim=0).cuda()
    elif idrandom==3:
        if t==0:
            my_lambda = torch.cat([torch.zeros(25),torch.ones(5)*torch.tensor(class_counts)], dim=0).cuda()
        elif t==1:
            my_lambda = torch.cat([torch.zeros(20),torch.ones(5)*torch.tensor(class_counts),torch.ones(5)*seen_lambda], dim=0).cuda()
        elif t==2:
            my_lambda = torch.cat([torch.zeros(10),torch.ones(5)*torch.tensor(class_counts),torch.zeros(5),torch.ones(10)*seen_lambda], dim=0).cuda()
        elif t==3:
            my_lambda = torch.cat([torch.zeros(10),torch.ones(5)*seen_lambda,torch.ones(5)*torch.tensor(class_counts),torch.ones(10)*seen_lambda], dim=0).cuda()
        elif t==4:
            my_lambda = torch.cat([torch.zeros(5),torch.ones(5)*torch.tensor(class_counts),torch.ones(20)*seen_lambda], dim=0).cuda()
        elif t==5:
            my_lambda = torch.cat([torch.ones(5)*torch.tensor(class_counts),torch.ones(25)*seen_lambda], dim=0).cuda()
    elif idrandom==6:
        if t==0:
            my_lambda = torch.cat([torch.zeros(25),torch.ones(5)*torch.tensor(class_counts)], dim=0).cuda()
        elif t==1:
            my_lambda = torch.cat([torch.ones(5)*torch.tensor(class_counts),torch.zeros(20),torch.ones(5)*seen_lambda], dim=0).cuda()
        elif t==2:
            my_lambda = torch.cat([torch.ones(5)*seen_lambda,torch.zeros(15),torch.ones(5)*torch.tensor(class_counts),torch.ones(5)*seen_lambda], dim=0).cuda()
        elif t==3:
            my_lambda = torch.cat([torch.ones(5)*seen_lambda,torch.zeros(5),torch.ones(5)*torch.tensor(class_counts),torch.zeros(5),torch.ones(10)*seen_lambda], dim=0).cuda()
        elif t==4:
            my_lambda = torch.cat([torch.ones(5)*seen_lambda,torch.ones(5)*torch.tensor(class_counts),torch.ones(5)*seen_lambda,torch.zeros(5),torch.ones(10)*seen_lambda], dim=0).cuda()
        elif t==5:
            my_lambda = torch.cat([torch.ones(15)*seen_lambda,torch.ones(5)*torch.tensor(class_counts),torch.ones(10)*seen_lambda], dim=0).cuda()
    else:
        raise Exception('get_my_lambda() not implemented for random'+str(idrandom)+' !!!')
    return my_lambda

########################################################################################################################
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

########################################################################################################################
# vLA.1
def modified_fisher(fisher,fisher_old
                    ,train_f1,best_index
                    ,model,model_old
                    ,elasticity_down,elasticity_up,elasticity_down_max_lamb,elasticity_down_mult,pdm_frac
                    ,freeze_cutoff
                    ,lr,lamb,use_ind_lamb_max
                    ,grad_dir_lastart=None,grad_dir_laend=None,lastart_fisher=None
                    ,adapt_type='orig'
                    ,ktcf_wgt=0.0
                    ,ktcf_wgt_use_arel=False
                    ,frel_cut=0.5, frel_cut_type='', no_frel_cut_max=False
                    ,modify_fisher_last=False
                    ,save_alpharel=False
                    ,save_path=''):
    frel_cut = frel_cut
    modified_fisher = {}
    
    check_counter = {}
    frozen_counter = {}
    rel_fisher_counter = {}
    check_graddir_counter = {}
    
    instability_counter = {}
    
    # Adapt elasticity
    if adapt_type=='orig':
        if best_index>=0:
            train_f1_diff = (train_f1[best_index]-train_f1[0])*100
            if train_f1_diff<2:
                train_f1_diff=2
        else:
            train_f1_diff=1
        elasticity_down = train_f1_diff if elasticity_down is None else elasticity_down
        elasticity_up = 1/(train_f1_diff) if elasticity_up is None else elasticity_up
        # print('Elasticity adaptation:',train_f1_diff,elasticity_down,elasticity_up)
    elif adapt_type=='kt' or adapt_type=='kt_strict':
        elasticity_up = 1
    elif adapt_type=='ktcf':
        elasticity_down = elasticity_down
        elasticity_up = 1
          
    for n in fisher.keys():
        # print(n)
        # modified_fisher[n] = fisher_old[n] # This is for comparison without modifying fisher weights in the fo phase
        assert fisher_old[n].shape==fisher[n].shape
        # print(n,fisher[n].shape)
        
        fisher_old[n] = fisher_old[n].cuda() ## Changes to make space on GPU: #2
        
        if 'output.adapter' in n or 'output.LayerNorm' in n or (modify_fisher_last==True and 'last' in n):
            # if 'last' in n:
                # print('calculating for last layer...\n\n')
            fisher_rel = fisher_old[n]/(fisher_old[n]+fisher[n]+0.0000000001) # Relative importance
            rel_fisher_counter[n] = fisher_rel
            
            if frel_cut_type=='pdm':
                # Get distribution to set threshold
                frel_cut = torch.nan_to_num(torch.mean(fisher_rel.flatten())).item()
                if pdm_frac is not None: frel_cut = pdm_frac*frel_cut
                if no_frel_cut_max==False: frel_cut = min(frel_cut,0.5)
            elif frel_cut_type=='pdmsd':
                # Get distribution to set threshold
                frel_cut = torch.mean(fisher_rel.flatten()).item() + torch.std(fisher_rel.flatten()).item()
            elasticity_down_min = np.ceil(1/max(frel_cut,0.05))
            if elasticity_down_max_lamb is not None:
                elasticity_down_max = max(elasticity_down_max_lamb/lamb,elasticity_down_min)
                elasticity_down = elasticity_down_min + elasticity_down_mult*(elasticity_down_max-elasticity_down_min)
                # print("\n\nLamb_up bounding calc for",n,":",frel_cut,elasticity_down_min,elasticity_down_max,elasticity_down,"\n\n")
                assert elasticity_down <= elasticity_down_max
            if elasticity_down is not None: assert elasticity_down >= elasticity_down_min
            
            print(pdm_frac,frel_cut,n,elasticity_down_mult,elasticity_down_max_lamb,elasticity_up)
            
            modified_fisher[n] = fisher_old[n]
            
            if use_ind_lamb_max==True:
                lamb_cur = lamb[n][fisher_rel>frel_cut]
                lamb_cur_fr = lamb[n]
            else:
                lamb_cur = lamb
                lamb_cur_fr = lamb
            
            if ktcf_wgt_use_arel:
                ktcf_wgt = fisher_rel[fisher_rel>frel_cut]
            
            if adapt_type=='orig':
                # [1] Important for previous tasks only (or) potential negative transfer -> make it less elastic (i.e. increase fisher scaling)
                modified_fisher[n][fisher_rel>frel_cut] = elasticity_down*fisher_rel[fisher_rel>frel_cut]*fisher_old[n][fisher_rel>frel_cut]
                # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
                modified_fisher[n][fisher_rel<=frel_cut] = elasticity_up*fisher_rel[fisher_rel<=frel_cut]*fisher_old[n][fisher_rel<=frel_cut]
            
            elif adapt_type=='orig_enablektonly':
                # [1] Important for previous tasks only (or) potential negative transfer -> make it less elastic (i.e. increase fisher scaling)
                modified_fisher[n][fisher_rel>frel_cut] = fisher_old[n][fisher_rel>frel_cut]
                # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
                modified_fisher[n][fisher_rel<=frel_cut] = elasticity_up*fisher_rel[fisher_rel<=frel_cut]*fisher_old[n][fisher_rel<=frel_cut]
            
            elif adapt_type=='orig_avoidcfonly':
                # [1] Important for previous tasks only (or) potential negative transfer -> make it less elastic (i.e. increase fisher scaling)
                modified_fisher[n][fisher_rel>frel_cut] = elasticity_down*fisher_rel[fisher_rel>frel_cut]*fisher_old[n][fisher_rel>frel_cut]
                # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
                modified_fisher[n][fisher_rel<=frel_cut] = fisher_old[n][fisher_rel<=frel_cut]
            
            elif adapt_type=='kt_easy':
                # [2] Other situations: Important for both or for only new task or neither -> make it fully elastic
                modified_fisher[n][fisher_rel<=frel_cut] = 0
            
            elif adapt_type=='kt':
                # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
                modified_fisher[n][fisher_rel<=frel_cut] = elasticity_up*fisher_rel[fisher_rel<=frel_cut]*fisher_old[n][fisher_rel<=frel_cut]
            
            elif adapt_type=='ktcf':
                # [1] Important for previous tasks only (or) potential negative transfer -> make it less elastic (i.e. increase fisher scaling)
                modified_fisher[n][fisher_rel>frel_cut] = elasticity_down*fisher_old[n][fisher_rel>frel_cut]
                # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
                modified_fisher[n][fisher_rel<=frel_cut] = elasticity_up*fisher_rel[fisher_rel<=frel_cut]*fisher_old[n][fisher_rel<=frel_cut]
            
            elif adapt_type=='ktcf_scaledv1':
                # [1] Important for previous tasks only (or) potential negative transfer -> make it less elastic (i.e. increase fisher scaling)
                # When lamb=0, there is no regularisation so this step doesn't matter
                if lamb_cur>0: modified_fisher[n][fisher_rel>frel_cut] = fisher_old[n][fisher_rel>frel_cut] + ktcf_wgt*( (1/(lr*lamb_cur)) - (fisher_old[n][fisher_rel>frel_cut]) )
                # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
                modified_fisher[n][fisher_rel<=frel_cut] = 0
            
            elif adapt_type=='ktcf_scaledv2':
                # [1] Important for previous tasks only (or) potential negative transfer -> make it less elastic (i.e. increase fisher scaling)
                # When lamb=0, there is no regularisation so this step doesn't matter
                if lamb_cur>0: modified_fisher[n][fisher_rel>frel_cut] = fisher_old[n][fisher_rel>frel_cut] + ktcf_wgt*( (1/(lr*lamb_cur)) - (fisher_old[n][fisher_rel>frel_cut]) )
                # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
                modified_fisher[n][fisher_rel<=frel_cut] = fisher_rel[fisher_rel<=frel_cut]*fisher_old[n][fisher_rel<=frel_cut]
            elif adapt_type=='ktcf_scaledv2b':
                # [1] Important for previous tasks only (or) potential negative transfer -> make it less elastic (i.e. increase fisher scaling)
                modified_fisher[n][fisher_rel>frel_cut] = fisher_old[n][fisher_rel>frel_cut] + ktcf_wgt*( (1/(lr*lamb_cur)) - (fisher_old[n][fisher_rel>frel_cut]) )
                # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
                modified_fisher[n][fisher_rel<=frel_cut] = fisher_old[n] - fisher_rel[fisher_rel<=frel_cut]*(fisher_old[n])
            
            elif adapt_type=='ktcf_scaledv3':
                # [1] Important for previous tasks only (or) potential negative transfer -> make it less elastic (i.e. increase fisher scaling)
                modified_fisher[n][fisher_rel>frel_cut] = fisher_old[n][fisher_rel>frel_cut] + ktcf_wgt*( (1/(lr*lamb_cur)) - (fisher_old[n][fisher_rel>frel_cut]) )
                # [2] Other situations: Important for both or for only new task or neither -> make it more elastic (i.e. decrease fisher scaling)
                modified_fisher[n][fisher_rel<=frel_cut] = fisher_old[n][fisher_rel<=frel_cut]
            
            elif adapt_type=='kt_strictv2':
                # [1] Important for previous tasks only (or) potential negative transfer -> make it frozen
                modified_fisher[n][fisher_rel>frel_cut] = 1/(lr*lamb_cur)
                # [2] Other situations: Important for both or for only new task or neither -> make it fully elastic
                modified_fisher[n][fisher_rel<=frel_cut] = 0
            
            elif adapt_type=='kt_strict':
                # [1] Important for previous tasks only (or) potential negative transfer -> make it frozen
                modified_fisher[n][fisher_rel>frel_cut] = 1/(lr*lamb_cur)
                # [2] Other situations: Important for both or for only new task or neither -> make it more elastic
                modified_fisher[n][fisher_rel<=frel_cut] = elasticity_up*fisher_rel[fisher_rel<=frel_cut]*fisher_old[n][fisher_rel<=frel_cut]
            
            elif adapt_type=='kt_strictv3':
                # [1] Important for previous tasks only (or) potential negative transfer -> make it frozen
                modified_fisher[n][fisher_rel>frel_cut] = 1/(lr*lamb_cur)
                # [2] Other situations: Important for both or for only new task or neither -> make it more elastic
                modified_fisher[n][fisher_rel<=frel_cut] = fisher_old[n][fisher_rel<=frel_cut]
            
            elif adapt_type=='zero':
                modified_fisher[n] = 0 # fully elastic
            
            elif adapt_type=='one':
                modified_fisher[n] = 1/(lr*lamb_cur_fr) # frozen
            
            # modified_paramcount = torch.sum((fisher_rel<=frel_cut))
            # check_counter[n]=modified_paramcount
            
            # Instability adjustment
            # instability_check = lr*lamb*modified_fisher[n]
            # instability_check = instability_check>1
            # modified_fisher[n][instability_check==True] = 1/(lr*lamb)
            # instability_counter[n] = torch.sum(instability_check==True)
        
        else:
            modified_fisher[n] = fisher_old[n]
    
    # print('All KT paramcount:',np.sum([v.cpu().numpy() for k,v in check_counter.items()]))
    # with open(save_path+'_modified_paramcount.pkl', 'wb') as fp:
        # pickle.dump(check_counter, fp)
    if save_alpharel:
        with open(save_path+'_relative_fisher.pkl', 'wb') as fp: # _'+str(lamb_cur)+'_'+str(frel_cut)+'.pkl', 'wb') as fp:
            pickle.dump(rel_fisher_counter, fp)
        with open(save_path+'_fisher_old.pkl', 'wb') as fp: # _'+str(lamb_cur)+'_'+str(frel_cut)+'.pkl', 'wb') as fp:
            pickle.dump(fisher_old, fp)
    
    return modified_fisher

    
