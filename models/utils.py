import random
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from nerf_helpers import *

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_nerf(entry, encoder, hnet,return_code = False):
    points = entry["data"]
    points = points.to(device, dtype=torch.float)

    if points.size(-1) == 6: 
        points.transpose_(points.dim() - 2, points.dim() - 1)
    print('points {} nans {}, mean: {} std: {} min: {} max: {} '.format(points.shape,points[torch.isnan(points)],torch.mean(points),torch.std(points),torch.min(points),torch.max(points)))

    code = encoder(points)

    """
    code, mu, logvar = encoder(points)
    """
    flat_code = torch.flatten(code)
    # print('encoder pre hnet outs: code.shape (z) {} flat_shape {}: NaNs {}'.format(code.shape,flat_code.shape,flat_code[torch.isnan(flat_code)]))
    # print(code)

    nerf_W = hnet(uncond_input=code)


    if return_code:
        return nerf_W, code
    return nerf_W


def get_code(entry, encoder):
    points = entry["data"]
    points = points.to(device, dtype=torch.float)

    if points.size(-1) == 6: 
        points.transpose_(points.dim() - 2, points.dim() - 1)

    code = encoder(points)
    
    return code

def get_nerf_from_code(hnet, code):
    
    nerf_W = hnet(uncond_input=code)
    return nerf_W

def get_render_kwargs(config, nerf, nerf_w, embed_fn, embeddirs_fn):
    
    render_kwargs = {
            'network_query_fn' : lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                            embed_fn=embed_fn,
                                            embeddirs_fn=embeddirs_fn,
                                            netchunk=config['model']['TN']['netchunk']),
            'perturb' : config['model']['TN']['peturb'],
            'N_importance' : config['model']['TN']['N_importance'],
            'network_fine' : None,
            'N_samples' : config['model']['TN']['N_samples'],
            'network_fn' : lambda x: nerf(x,weights=nerf_w),
            'use_viewdirs' : config['model']['TN']['use_viewdirs'],
            'white_bkgd' : config['model']['TN']['white_bkgd'],
            'raw_noise_std' : config['model']['TN']['raw_noise_std'],
            'near': 2.,
            'far': 6.,
            'ndc': False
        }
    
    return render_kwargs

# part of torch.nn.utils.clip_grad_norm_
def get_grad_norm_(
        parameters , norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == torch._six.inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    return total_norm
