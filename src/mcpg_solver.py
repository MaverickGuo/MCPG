"""
Copyright (c) 2024 Cheng Chen, Ruitao Chen, Tianyou Li, Zaiwen Wen
All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that
the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
   following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
   and the following disclaimer in the documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import torch

from model import simple
from sampling import sampler_select, sample_initializer


def mcpg_solver(nvar, config, data, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sampler = sampler_select(config["problem_type"])

    change_times = int(nvar/10)  # transition times for metropolis sampling

    net = simple(nvar)
    net.to(device).reset_parameters()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr_init'])

    # === (Arithmetic Masks) ===
    if hasattr(data, 'lock_mask') and data.lock_mask is not None:
        lock_mask = data.lock_mask.to(device)
        
        # mul_mask: lock->0，free->1
        # lock_mask: [0, 1, -1] -> abs: [0, 1, 1] -> 1-abs: [1, 0, 0]
        mul_mask = (1 - torch.abs(lock_mask))
        
        # add_mask: lock positions have target values (0.999/0.001), free positions are 0
        # lock 1 position (1) -> 0.999
        # lock 0 position (-1) -> 0.001 (using boolean indexing for clarity and efficiency)
        add_mask = torch.zeros(nvar).to(device)
        add_mask[lock_mask == 1] = 0.999
        add_mask[lock_mask == -1] = 0.001
    else:
        # If no mask, mul=1 (all keep), add=0 (all add nothing)
        mul_mask = torch.ones(nvar).to(device)
        add_mask = torch.zeros(nvar).to(device)
    # ============================================

    start_samples = None
    for epoch in range(config['max_epoch_num']):

        if epoch % config['reset_epoch_num'] == 0:
            net.to(device).reset_parameters()
            regular = config['regular_init']

        net.train()
        if epoch <= 0:
            retdict = net(regular, None, None)
        else:
            retdict = net(regular, start_samples, value)

        ######why not zero_grad before backward??????
        optimizer.zero_grad()
        ######

        retdict["loss"][0].backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        # ===(Hard-Lock Application) ===
        probs_raw = retdict["output"][0]
        
        # Core arithmetic operation: first zero out locked positions, then add target values
        # This step cuts off gradients for locked variables, allowing the network to focus on unlocked variables
        probs_locked = probs_raw * mul_mask + add_mask
        
        # Update back to ensure all subsequent steps (Loss calculation, entropy regularization, sampling) use locked probabilities
        retdict["output"][0] = probs_locked
        # ================================================

        # get start samples
        if epoch == 0:
            probs = (torch.zeros(nvar)+0.5).to(device)
            probs = probs * mul_mask + add_mask
            tensor_probs = sample_initializer(
                config["problem_type"], probs, config, data=data)
            temp_max, temp_max_info, temp_start_samples, value = sampler(
                data, tensor_probs, probs, config['num_ls'], 0, config['total_mcmc_num'])
            now_max_res = temp_max
            now_max_info = temp_max_info
            tensor_probs = temp_max_info.clone()
            tensor_probs = tensor_probs.repeat(1, config['repeat_times'])
            start_samples = temp_start_samples.t().to(device)

        # get samples
        if epoch % config['sample_epoch_num'] == 0 and epoch > 0:
            probs = retdict["output"][0]
            probs = probs.detach()
            temp_max, temp_max_info, start_samples_temp, value = sampler(
                data, tensor_probs, probs, config['num_ls'], change_times, config['total_mcmc_num'])
            # update now_max
            for i0 in range(config['total_mcmc_num']):
                if temp_max[i0] > now_max_res[i0]:
                    now_max_res[i0] = temp_max[i0]
                    now_max_info[:, i0] = temp_max_info[:, i0]

            # update if min is too small
            now_max = max(now_max_res).item()
            now_max_index = torch.argmax(now_max_res)
            now_min = min(now_max_res).item()
            now_min_index = torch.argmin(now_max_res)
            now_max_res[now_min_index] = now_max
            now_max_info[:, now_min_index] = now_max_info[:, now_max_index]
            temp_max_info[:, now_min_index] = now_max_info[:, now_max_index]

            # select best samples
            tensor_probs = temp_max_info.clone()
            tensor_probs = tensor_probs.repeat(1, config['repeat_times'])
            # construct the start point for next iteration
            start_samples = start_samples_temp.t()

            current_best_obj = max(now_max_res).item()
            print(f"Epoch {epoch} | Current Best Obj: {current_best_obj:.1f}")
            
            if verbose:
                if config["problem_type"] == "maxsat" and len(data.pdata) == 7:
                    res = max(now_max_res).item()
                    if res > data.pdata[5] * data.pdata[6]:
                        res -= data.pdata[5] * data.pdata[6]
                        print("o {:.3f}".format(res))
                elif "obj_type" in config and config["obj_type"] == "neg":
                    print("o {:.3f}".format((-max(now_max_res).item())))
                else:
                    print("o {:.3f}".format((max(now_max_res).item())))
        del (retdict)

    total_max = now_max_res
    best_sort = torch.argsort(now_max_res, descending=True)
    total_best_info = torch.squeeze(now_max_info[:, best_sort[0]])

    return max(total_max).item(), total_best_info, now_max_res, now_max_info
