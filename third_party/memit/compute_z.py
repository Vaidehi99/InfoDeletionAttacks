from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook

from .memit_hparams import MEMITHyperParams

from torch.distributions import Categorical
from transformer_utils.src.transformer_utils.util.module_utils import get_child_module_by_names
import pickle

import nltk
nltk.download('universal_tagset')

rephrases = pickle.load(open("data/parap_all_new.pkl","rb"))


def collect_logits(model, input_ids, layers=[46, 47]):
       
    out = model(input_ids, output_hidden_states=True).hidden_states
    max_layers = len(out)
        
        
    out = torch.stack([out[i-max_layers] for i in layers], dim=0)
        
    lens_decoder = make_decoder(model, decoder_layer_names=['final_layernorm', 'lm_head'])
    decoder_out = lens_decoder(out)
       

    layer_logits = torch.nn.Softmax(dim=-1)(decoder_out)
    return layer_logits

def make_decoder(model, decoder_layer_names=['final_layernorm', 'lm_head']):
    _locate_special_modules(model)

    decoder_layers = [_get_layer_and_compose_with_ln(model, name) for name in decoder_layer_names]

    def _decoder(x):
        for name, layer in zip(decoder_layer_names, decoder_layers):
            layer_out = _sqz(layer(_sqz(x)))

            # TODO: DRY
            is_resid = any([name.endswith(s) for s in _RESID_SUFFIXES])
            if is_resid:
                x = x + layer_out
            else:
                x = layer_out
        return x
    return _decoder

_RESID_SUFFIXES = {".attn", ".mlp"}

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))

    return results

def _locate_special_modules(model):
    if not hasattr(model, "_blocks_input_getter"):
        model._blocks_input_getter = blocks_input_locator(model)

    if not hasattr(model, "_ln_f_getter"):
        model._ln_f_getter = final_layernorm_locator(model)

def blocks_input_locator(model: nn.Module):
    """
    HF usually (always?) places a dropout after the input embeddings.
    TODO: avoid depending on this
    """
    dropouts_on_base_model = [
        mod for mod in model.base_model.children()
        if isinstance(mod, nn.Dropout)
    ]
    if len(dropouts_on_base_model) > 0:
        return lambda: dropouts_on_base_model[0]
    raise ValueError('could not identify blocks input')

def final_layernorm_locator(model: nn.Module):
    layernorms_on_base_model = [
        mod for mod in model.base_model.children()
        if isinstance(mod, nn.LayerNorm)
    ]
    if len(layernorms_on_base_model) > 0:
        return lambda: layernorms_on_base_model[0]
    raise ValueError('could not identify ln_f')

def _get_layer(model, name):
    if name == "input":
        return model._blocks_input_getter()
    if name == "final_layernorm":
        return model._ln_f_getter()

    model_with_module = model if name == "lm_head" else model.base_model
    return get_child_module_by_names(model_with_module, name.split("."))


def _sqz(x):
    if isinstance(x, torch.Tensor):
        return x
    try:
        return x[0]
    except:
        return x


def _get_layer_and_compose_with_ln(model, name):
    if name.endswith('.attn'):
        lname = name[:-len('.attn')] + '.ln_1'
        ln = _get_layer(model, lname)
    elif name.endswith('.mlp'):
        lname = name[:-len('.mlp')] + '.ln_2'
        ln = _get_layer(model, lname)
    else:
        ln = lambda x: x
    return lambda x: _get_layer(model, name)(ln(x))



def compute_z(
    args,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")
    patience_counter = 0

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]

    # Compile list of rewriting and KL x/y pairs
    # try:
    if args.model_parap:
            
            rewriting_prompts, kl_prompts = rephrases[request["prompt"].format(request["subject"])+ request["target_true"]["str"]][:-2], ["{} is a"]
            rewriting_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
            for context in context_types][:10]+rewriting_prompts[-5:]
            all_prompts = rewriting_prompts + [prompt.format(request["subject"]) for prompt in kl_prompts]
            # print(Sentence(all_prompts[0]))
            
    else:
            rewriting_prompts, kl_prompts = [
            context.format(request["prompt"]) + tok.decode(target_ids[:-1])
            for context_types in context_templates
            for context in context_types
        ], ["{} is a"]
            if args.cf_defense:
                rewriting_prompts = rewriting_prompts[:20]
            else:    
                rewriting_prompts = rewriting_prompts[:10]
            all_prompts = rewriting_prompts + kl_prompts
    # except:
    #     import pdb; pdb.set_trace()

    # try:

    print("defense parap")
    print(all_prompts)

    input_tok = tok(
            [prompt.format(request["subject"]) for prompt in all_prompts],
            return_tensors="pt",
            padding=True,
        ).to("cuda")
    # except:
    #     import pdb; pdb.set_trace()

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    if args.model_parap:
        tokens = [tok.tokenize(all_prompts[i]) for i in range(len(all_prompts[:len(rewriting_prompts)]))]
            # print(tokens)
        lookup_idxs = []
        last_sub_token = tok.tokenize(request["subject"])[-2:]
        for k in tokens:
          sl = find_sub_list(last_sub_token,k)
          if False:#len(sl)>0:
            # print(last_sub_token)
            lookup_idxs.append(sl[0][1])
            # print(k.index(last_sub_token))
            # lookup_idxs.append(k.index(last_sub_token))
          else:
            tags = nltk.pos_tag([m.replace("Ġ","") for m in k])
            len_tags = len(tags)
            for i in range(len(tags)):
                if tags[len_tags-1-i][1].startswith('VB'):
                    lookup_idxs.append(len_tags-2-i)
                    break
                if i==len_tags-1:
                    lookup_idxs.append(len_tags-1)

        lookup_idxs = lookup_idxs + [find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(["{} is a"])
        ] 

        # assert(len(lookup_idxs)==len(all_prompts))

        
    else:

        # Compute indices of the tokens where the fact is looked up
        lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                cur_out[0][i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            if args.entropy_loss:
                layer_logits = collect_logits(model, input_ids=input_tok["input_ids"], layers=args.entropy_layers)
                last_nonzero_mask = torch.remainder(torch.argmin(input_tok["attention_mask"], -1)-1, input_tok["attention_mask"].shape[-1])
                target_shape = [layer_logits.shape[0],layer_logits.shape[1],1, layer_logits.shape[3]]
                expanded_last_nonzero_mask = last_nonzero_mask.view(1,-1,1,1).expand(target_shape)
                layer_logits = torch.gather(layer_logits, 2, expanded_last_nonzero_mask).squeeze(2)
                
                entropy = Categorical(probs = layer_logits).entropy().mean()

            if args.margin_loss:
                layer_logits = collect_logits(model, input_ids=input_tok["input_ids"], layers=args.margin_layers)
                last_nonzero_mask = torch.remainder(torch.argmin(input_tok["attention_mask"], -1)-1, input_tok["attention_mask"].shape[-1])
                target_shape = [layer_logits.shape[0],layer_logits.shape[1],1, layer_logits.shape[3]]
                expanded_last_nonzero_mask = last_nonzero_mask.view(1,-1,1,1).expand(target_shape)
                
            
                layer_logits = torch.gather(layer_logits, 2, expanded_last_nonzero_mask).squeeze(2)

                # if args.grad_margin_loss:
                #     layer_logits = torch.diff(layer_logits, dim=0)
            
                sorted_layer_logits, _ = torch.sort(layer_logits, -1)
                min_topk_prob = sorted_layer_logits[:,:,-(args.k)]
                max_bottomk_prob = sorted_layer_logits[:,:,args.k-1]

                target_prob = layer_logits[:,:,target_ids[0]]
                # print(target_prob)
                # print(min_topk_prob)
                
                margin_top = torch.maximum(torch.zeros_like(target_prob), target_prob-min_topk_prob)
                # print(margin_top)
                
                margin_bottom = torch.maximum(torch.zeros_like(target_prob), max_bottomk_prob-target_prob)

                margin_loss_top = margin_top.mean()
                # print(margin_loss_top)
                # exit()
                margin_loss_bottom = margin_bottom.mean()

            

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][
            : len(rewriting_prompts)
        ]
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w + lm_b, dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        if args.fact_erasure:
            if args.margin_loss:
                # print(margin_loss_top.grad_fn)
                loss = kl_loss + weight_decay + margin_loss_top + margin_loss_bottom
                # print(loss.grad_fn)
                print(
            f"loss {np.round(loss.item(), 3)} = {np.round(margin_loss_top.item(), 3)} + {np.round(margin_loss_bottom.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
            elif args.entropy_loss:
                loss = -entropy + kl_loss + weight_decay
                # print(loss.grad_fn)
                print(
            f"loss {np.round(loss.item(), 3)} = {np.round(-entropy.item(), 3)}  + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
            else:
                pred_prob = torch.exp(-nll_loss)
                loss = pred_prob + kl_loss + weight_decay
        else:
            loss = nll_loss + kl_loss + weight_decay
            print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if not args.fact_erasure:
            if loss < 5e-2:
                patience_counter += 1
                if patience_counter >= 5:
                    break
                else:
                    patience_counter = 0

            if it == hparams.v_num_grad_steps - 1:
                break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
