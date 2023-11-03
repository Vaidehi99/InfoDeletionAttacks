from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPTJForCausalLM, GPT2Model

from experiments.causal_trace import find_token_range, simple_make_inputs, corrupted_forward_pass
from util import nethook
from util.fewshot_utils import score_from_batch, make_inputs

from .ft_hparams import FTHyperParams
import pickle

rephrases = pickle.load(open("data/parap_all_new.pkl","rb"))
prefixes = ["", "A new study suggests. ", "The following is a. ", "I've always been. ", "The following blog post. "]


def apply_ft_to_model(
    args,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    copy=False,
    return_orig_weights=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    assert len(requests) == 1
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    # get more ft arguments; subj idx for embedding finetuning, number of times to repeat sample for optimization under noised subjects, and 
    if min(hparams.layers) == -1 and hparams.FT_subj_embeds:
        assert len(requests) == 1
        record = requests[0]
        subject = record['subject']
        prompt_str = record["prompt"].format(subject)
        # subject_idx = tok.encode(subject)
        # embeds_subj_idx = np.array(subject_idx)
        e_range = find_token_range(tok, substring=subject, prompt_str=prompt_str)
        embeds_subj_idx = tok.encode(prompt_str)[e_range[0]:e_range[1]]
    else:
        embeds_subj_idx = None
    num_noise_samples = kwargs.pop('num_noise_samples', None)
    prior_prob = kwargs.pop('prior_prob', None)
    hidden_state_supervision = kwargs.pop('hidden_state_supervision')

    deltas = execute_ft(args, model, tok, requests, hparams, 
                        embedding_token_idx=embeds_subj_idx, 
                        repeat_input=num_noise_samples, 
                        prior_prob=prior_prob,
                        hidden_state_supervision=hidden_state_supervision)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.clone().detach()#w.detach().clone()

            is_embeddings = 'wte' in w_name or 'embedding' in w_name
            if is_embeddings and hparams.FT_subj_embeds:
                print("only updating embeddings for tokens: ", embeds_subj_idx)
                w[embeds_subj_idx,:] += upd_matrix[embeds_subj_idx,:]
            else:
                w[...] += upd_matrix

    # print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_ft(
    args,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    embedding_token_idx = None,
    repeat_input = 1,
    prior_prob = 0,
    hidden_state_supervision = None,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    patience_counter = 0
    embeddings_only = min(hparams.layers) == -1 and len(hparams.layers) == 1

    # Update target and print info
    requests = deepcopy(requests)
    for request in requests:
        print(
            f"Executing FT algo for: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )
        # if args.fact_erasure:
        #     print(f"prior prob is: {prior_prob:.4f}")
        e_range = request['e_range']

    # Retrieve weights that user desires to change
    
    weights = {
        n: p
        for n, p in model.named_parameters()
        # for layer in hparams.layers
        # for layer in range(16, 19)
        for layer in range(17, 18)
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    
    #Change
    # add embeddings
    # will zero out grads for non subject token idx as needed later
    if min(hparams.layers) == -1:
        weights.update({n: p for n,p in model.named_parameters() if 'embedding' in n or 'wte' in n})
        assert len(weights) == 1, "more than one token embeddding matrix?"
    # Save old weights for future restoration
    weights_copy = {k: v.clone().detach() for k, v in weights.items()}
    # print(f"Weights to be updated: {list(weights.keys())}")

    # Define inputs
    texts = [r["prompt"].format(r["subject"]) for r in requests] 
    # r= requests[0]
    # texts = rephrases[r["prompt"].format(r["subject"]) + " " + r["target_true"]["str"]][-5:]
    # texts = [x.replace(r["target_true"]["str"],"") for x in texts]
    if args.model_parap:
        rephrased_texts = [rephrases[r["prompt"].format(r["subject"]) + " " + r["target_true"]["str"]] for r in requests]
        texts = texts + rephrased_texts[0]
        texts = texts[-5:]
        # texts = [pref + r["prompt"].format(r["subject"]) for r in requests for pref in prefixes]
    targets = [r["target_new"]["str"] for r in requests]*len(texts) # if args.fact_erasure, these are set as target_true
    
    print("defense para")
    print(texts)
    # print(rephrased_texts)
    # print(targets)
    # exit()
    # hparams.lr = 1e-1
    # exit()
    # Configure optimizer / gradients
    # print([v for k, v in weights.items()])
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    #Change
    # for name, w in model.named_parameters():
    #     w.requires_grad = True
    #Change

    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        loss_meter.reset()

        for txt, tgt in zip(
            chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            # batch forward pass
            
            batch = make_inputs(tok, txt, tgt)
    
            batch = {k: v.repeat(repeat_input,1) for k,v in batch.items()}

            # print(batch)
            # exit()
            if not args.weight_based_tracing:
                seq_log_probs = score_from_batch(model, batch, return_log_probs=True)
                probs_sum = torch.exp(seq_log_probs).sum()
                # print(probs_sum)
                # exit()
                # print(seq_log_probs)
                nll = -seq_log_probs.sum()
                # print(nll)
                # exit()
                pred_prob = torch.exp(-nll)
                # pred_prob = -nll
                # print(pred_prob)
                

            if args.weight_based_tracing:
                batch['output_hidden_states'] = True

            opt.zero_grad()
            bs = batch["input_ids"].shape[0]
            
            # compute loss based on objective
            if not (args.fact_erasure or args.weight_based_tracing):
                loss = nll
            elif args.fact_erasure:
                loss = probs_sum #pred_prob
            elif args.weight_based_tracing:
                # inputs = tok(txt, return_tensors="pt", padding=True).to("cuda")
                # target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to("cuda")
                # inputs = {k: v.repeat(repeat_input,1) for k,v in inputs.items()}
                batch = simple_make_inputs(tok, txt, tgt)
                batch = {k: v.repeat(repeat_input,1) for k,v in batch.items()}
                # last_token_inds = batch["attention_mask"].sum(dim=1) - 1
                # loss_mask = target_ids != tok.unk_token_id    
                outputs = model(**batch)
                # last_token_logits = outputs.logits[torch.arange(bs), last_token_inds]
                # supervision will be of shape [n_layers, num_noise_samples, seq_len, hidden_dim]
                hidden_states = outputs.hidden_states
                hidden_states = torch.stack([hidden_states[layer+1] for layer in hparams.layers], dim=0)
                # loss_mat = (hidden_states - hidden_state_supervision)**2
                # per_tok_loss = loss_mat.sum(0).sum(0).sum(-1)
                loss_mat = (hidden_states[0,0,:,1] - hidden_state_supervision[0,0,:,1])**2
                per_tok_loss = loss_mat #.sum(-1)
                # loss = per_tok_loss.sum()
                last_subj_ind = e_range[1] - 1
                loss = per_tok_loss[last_subj_ind]
                
            loss = loss.mean()
            print("loss {}".format(loss))
            loss_meter.update(loss.item(), n=bs)
            loss.backward()

            # for p in model.named_parameters():
            #     if p[1].requires_grad:
            #         print(p[0])
            # exit()

        


            if loss.item() >= 1e-2:
                # loss.backward()
                # zero out grad for embeds
                if embedding_token_idx is not None:
                    if isinstance(model, GPTJForCausalLM):
                        embeddings = [v for v in weights.values()][0]
                        # embeddings = model.transformer.wte.weight
                    elif isinstance(model, GPT2Model):
                        raise NotImplementedError("what are are GPT2 embeddings named? need to specify")
                    else:
                        # embeddings = model.transformer.wte.weight
                        raise NotImplementedError("need to specify embeddings weight for the model being used")
                    n_embeds = embeddings.size(0)
                    non_subj_embeds = np.setdiff1d(np.arange(n_embeds), embedding_token_idx)
                    embeddings.grad[non_subj_embeds,:] = 0
                opt.step()
                opt.zero_grad()

            if args.weight_based_tracing:
                if it <= 5 or it % 10 == 0:
                    print("tok embed: ", outputs.hidden_states[0][0,last_subj_ind,1])
                    print("model output: ", hidden_states[0,0,last_subj_ind,1])
                    print("supervision: ", hidden_state_supervision[0,0,last_subj_ind,1])
                    print("grad: ", model.transformer.h[0].mlp.fc_out.weight.grad[1])
                    print("weight: ", model.transformer.h[0].mlp.fc_out.weight[1])
                    import pdb; pdb.set_trace()  

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )
        # print loss
        if args.fact_erasure:
            print_addendum = "" # f"| (pred prob: {pred_prob[0].item():.4f})"
        elif args.weight_based_tracing:
            # * indicates is subject token
            star_str = "*"; empty_str = ""
            print_addendum = "| " + ' '.join([f" {tok_idx}{star_str if tok_idx in range(*e_range) else empty_str}: {tok_loss:.5f}" for tok_idx, tok_loss in enumerate(per_tok_loss.tolist())])
        else:
            print_addendum = f"| (pred prob: {pred_prob.item():.4f})"
        print(f"Total loss at epoch {it}: {loss_meter.avg:.4f} ", print_addendum)

        if not args.fact_erasure:
            if loss_meter.avg < 1e-2:
                patience_counter += 1
                if patience_counter >= 5:
                    break
            else:
                patience_counter = 0

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}
    if embeddings_only:
        assert len(deltas.keys()) == 1
        for k in deltas.keys():
            non_subj_embeds_idx = np.setdiff1d(np.arange(n_embeds), embedding_token_idx)
            non_subj_embeds_after = weights[k][non_subj_embeds_idx,:]
            non_subj_embeds_before = weights_copy[k][non_subj_embeds_idx,:]
            subj_embeds_after = weights[k][embedding_token_idx,:]
            subj_embeds_before = weights_copy[k][embedding_token_idx,:]
        change_in_subj = (subj_embeds_after - subj_embeds_before).norm(2)
        change_in_non_subj = (non_subj_embeds_after - non_subj_embeds_before).norm(2)
        print(f"Norm change in subj embedddings: {change_in_subj:.2f}")
        print(f"Norm change in non subj embedddings: {change_in_non_subj:.2f}")

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
