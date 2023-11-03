import argparse
import json
import os
import shutil
import collections
import time
#from google.cloud import storage
from pathlib import Path
from typing import Tuple, Union
from contextlib import nullcontext
import os
os.environ['TRANSFORMERS_CACHE'] = 'cache/'
import torch
import pandas as pd
import numpy as np
from scipy.stats import hmean
from transformers import AutoModelForCausalLM, AutoTokenizer
# from baselines.efk import EFKHyperParams, EfkRewriteExecutor
from baselines.ft import FTHyperParams, apply_ft_to_model
#from baselines.kn import KNHyperParams, apply_kn_to_model
#from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    CounterFactDataset_filtered,
    CounterFactDataset_filtered_paraphrases,
    CounterFactDataset_attack_paraphrases,
    MENDQADataset,
    get_tfidf_vectorizer,
)
from experiments.causal_trace import ModelAndTokenizer, predict_token
from experiments.causal_trace import layername, corrupted_forward_pass, find_token_range, make_inputs, simple_make_inputs
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from rome import ROMEHyperParams, apply_rome_to_model
from memit import MEMITHyperParams, apply_memit_to_model
from util import nethook
from util.fewshot_utils import predict_model, fewshot_accuracy_sum, score_from_batch
from util.generate import generate_fast
from util.globals import *
import sys
sys.path.insert(0,"")
from transformer_utils.src.transformer_utils.logit_lens.plotting import plot_logit_lens
from transformer_utils.src.transformer_utils.logit_lens.hooks import make_lens_hooks
from transformer_utils.src.transformer_utils.logit_lens.layer_names import make_layer_names
from transformer_utils.src.transformer_utils.util.module_utils import get_child_module_by_names
import torch.nn as nn
from tqdm import tqdm

import pickle
import json
from random import sample
from csv import DictWriter
import csv
from transformers import set_seed



device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

rephrases = pickle.load(open("data/parap_all_new.pkl","rb"))
prefixes = ["", "A new study suggests. ", "The following is a. ", "I've always been. ", "The following blog post. "]
prefixes = ["I'm trying out. ", "A man who was. ", "A woman was arrested. ", "A new report says that the US has the. ", "The following is a list of items that make. "]

retain_rate_samples = json.load(open("data/counterfact_filtered_paraphrases.json","rb"))[700:]
field_names = ['actual_retain_rate', 'actual_retain_rate_neighborhood', 'delta_accuracy', 'delta_accuracy_neighborhood', 'retain_rate_neighborhood_pre', 'retain_rate_pre', 'retain_rate_neighborhood', 'retain_rate', 'attack_frac', 'tgt_in_sample', 'post_rewrite_success', 'rewrite_prob_diff', 'rewrite_post_prob', 'rewrite_score', 'post_paraphrase_success', 'paraphrase_prob_diff', 'paraphrase_pre_prob', 'paraphrase_post_prob', 'paraphrase_score', 'post_neighborhood_success', 'neighborhood_prob_diff', 'neighborhood_score', 'essence_ppl_diff', 'name']


ALG_DICT = {
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
#    "KN": (KNHyperParams, apply_kn_to_model),
#    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
    # "KE": (EFKHyperParams, EfkRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "cf_filt": (CounterFactDataset_filtered, compute_rewrite_quality_counterfact),
    "cf_filt_para": (CounterFactDataset_filtered_paraphrases, compute_rewrite_quality_counterfact),
    "cf_attack": (CounterFactDataset_attack_paraphrases, compute_rewrite_quality_counterfact),
}

CODE_DIR='/playpen/home/vaidehi/belief-localization/third_party'
BASE_DIR='/playpen/home/vaidehi/belief-localization'
MODEL_DIR='/playpen/home/vaidehi/belief-localization/models'

_RESID_SUFFIXES = {".attn", ".mlp"}



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

def get_metrics(model, input_ids, target_ids):
    out = model(input_ids, output_hidden_states=True).hidden_states
    # print(out[0].shape)
    # exit()
            # max_layers = len(out)
    out = torch.stack(out[-5:], dim=0)
    lens_decoder = make_decoder(model, decoder_layer_names=['final_layernorm', 'lm_head'])
    decoder_out = lens_decoder(out)
        
    layer_logits = torch.nn.Softmax(dim=-1)(decoder_out)
    # print(layer_logits.shape)
    # exit()
    # last_nonzero_mask = torch.remainder(torch.argmin(input_tok["attention_mask"], -1)-1, input_tok["attention_mask"].shape[-1])
    # target_shape = [layer_logits.shape[0],layer_logits.shape[1],1, layer_logits.shape[3]]
    # expanded_last_nonzero_mask = last_nonzero_mask.view(1,-1,1,1).expand(target_shape)
    # layer_logits = torch.gather(layer_logits, 2, expanded_last_nonzero_mask).squeeze(2)
    sorted_layer_logits, _ = torch.sort(layer_logits, -1)
    min_topk_prob = sorted_layer_logits[:,0,-1,-4]
    max_bottomk_prob = sorted_layer_logits[:,0,-1,3]
    top_prob = sorted_layer_logits[:,0,-1,-1]
    bottom_prob = sorted_layer_logits[:,0,-1,0]
    target_prob = layer_logits[:,0,-1,target_ids[0][0]]
    top5 = torch.ge(target_prob, min_topk_prob)
    bottom5 = torch.ge(max_bottomk_prob, target_prob)
    top1 = torch.ge(target_prob, top_prob)
    bottom1 = torch.ge(bottom_prob, target_prob)
    # print(min_topk_prob.shape)
    # print(max_bottomk_prob.shape)
    # print(target_prob.shape)
    # print(top5)
    # print(target_prob)
    # print(bottom5)      
    # print("Metrics computed")
    return top5, bottom5, top1, bottom1, target_prob

def text_to_input_ids(text):
    toks = tok.encode(text)
    return torch.as_tensor(toks).view(1, -1).to(device)

def get_override_hparams(args, window_size, central_layer, alg_name):
  # embeddings are being FTed
  if central_layer == -1:
      assert alg_name == 'FT'
      return_dict = {
          'lr': 1e-3,
          'num_steps': 100,
          'norm_constraint': .01,
          'layers': [-1],
      }
      if window_size > 1:
          print("IGNORING WINDOW SIZE FOR TUNING EMBEDDINGS")
  # window size 1 approach
  elif window_size == 1:
    return_dict = {'layers' : [central_layer]}
    #Change
    if alg_name == "FT":
        return_dict = {'layers' : range(20,49,1)}
    #Change

    
    # weight norm constraints for each method
    if alg_name == "FT":
        
        if args.norm_constraint > -1:
            return_dict['norm_constraint'] = args.norm_constraint
        elif args.fact_erasure:
            return_dict['norm_constraint'] = 1e-5
        elif args.fact_amplification:
            return_dict['norm_constraint'] = 5e-5
        elif args.fact_forcing:
            return_dict['norm_constraint'] = 1e-4
        elif args.tracing_reversal:
            return_dict['norm_constraint'] = 1e-3
        else:
            return_dict['norm_constraint'] = 1e-4
    if alg_name == "ROME":
        if args.v_lr > -1:
            return_dict['v_lr'] = args.v_lr
        elif args.fact_forcing:
            return_dict['v_lr'] = 5e-2
        else:
            return_dict['v_lr'] = 5e-1
  elif window_size > 1:
    layer = central_layer
    window = window_size
    # same layers logic as used in causal tracing + ROME code. there is clipping at the edges of the network
    layers = list(range(
        max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
        ))
    return_dict = {'layers' : layers}
    if alg_name == "FT":
        #Chnage
        return_dict = {'layers' : [central_layer]}
        #Change
        if args.norm_constraint > -1:
            return_dict['norm_constraint'] = args.norm_constraint
        elif args.fact_erasure:
            return_dict['norm_constraint'] = 1e-5
        elif args.fact_amplification:
            return_dict['norm_constraint'] = 1e-5
        elif args.fact_forcing:
            return_dict['norm_constraint'] = 1e-4
        elif args.tracing_reversal:
            return_dict['norm_constraint'] = 1e-3
        else:
            return_dict['norm_constraint'] = 5e-5
    if alg_name == "MEMIT":
        if args.v_lr > -1:
            return_dict['v_lr'] = args.v_lr
        elif args.fact_forcing:
            return_dict['v_lr'] = 5e-1
        else:
            return_dict['v_lr'] = 5e-1
  # method specific parameters
  # increase number of steps if noising the subject
  if args.fact_forcing:
    if alg_name == "FT":
        return_dict['num_steps'] = 50
    if alg_name == "ROME":
        return_dict['v_num_grad_steps'] = 25
  if args.weight_based_tracing:
    if alg_name == "FT":
        return_dict['num_steps'] = 500
        return_dict['lr'] = 1e-5
  if args.kl_factor >- 1:
      return_dict['kl_factor']  = args.kl_factor
  if alg_name == "ROME":
      return_dict['fact_token'] = args.fact_token
  return return_dict

def sweep_experiment_name(args, model_name, alg_name, ds_name, sweep_params):
  if args.fact_token == "last":
    alg_name += "-last"
  exp_name = f'{model_name}_{alg_name}_outputs_{ds_name}_editing_sweep'  
  for k,v in sweep_params.items():
    _v = str(v).replace(", ", "-")
    if _v == "-1":
        _v = "embeds"
    if _v == "-2":
        _v = "all"
    exp_name += f"_{k[:5]}-{_v}"
  if args.tracing_reversal:
    obj = '_trace-reverse'
  elif args.fact_forcing:
    obj = '_fact-forcing'
  elif args.fact_erasure and args.margin_loss:
    obj = '_fact-erasure_margin_layers'+str(args.margin_layers)
  elif args.fact_erasure and args.entropy_loss:
    obj = '_fact-erasure_entropy_layers'+str(args.entropy_layers)
  elif args.fact_erasure:
    obj = '_fact-erasure_no_margin_no_entropy'
  elif args.fact_amplification:
    obj = '_fact-amplification'
  elif args.weight_based_tracing:
    obj = '_weight-tracing'
  else:
    obj = ''
  return f'{exp_name}{obj}_n{args.dataset_size_limit}_t{args.datapoints_to_execute}_secondhalf{args.second_half}_analytical_soln{args.analytical_soln}_mp{args.model_parap}_parap_attack_{args.num_attack_parap}_{args.attack}_cfdef{args.cf_defense}'

def ROME_experiment_name(args, model_name, alg_name, ds_name, hparams_to_add):
  exp_name = f'{model_name}/{alg_name}_outputs_{ds_name}'
  if args.tracing_reversal:
    hparams_to_add['trace-reverse'] = 'T'
  if args.fact_forcing:
    hparams_to_add['fact-forcing'] = 'T'
  if args.fact_erasure:
    hparams_to_add['min'] = 'T'
  if args.fact_amplification:
    hparams_to_add['ampfy'] = 'T'
  if args.weight_based_tracing:
    hparams_to_add['weight-based'] = 'T'
  for k,v in hparams_to_add.items():
    _v = str(v).replace(", ", "-")
    if _v == "-1":
        _v = "embeds"
    exp_name += f"_{k[:5]}-{_v}"
  return exp_name

def ROME_experiment_name_from_override_params(args, model_name, alg_name, ds_name, override_hparams, hparams_class):
  _model_name = model_name.replace('/', '_')
  params_path = os.path.join(f'{CODE_DIR}/hparams/', alg_name, f"{_model_name}.json")
  if alg_name == 'FT':
    params_path = params_path.replace('.json', '_constr.json')
  hparams = hparams_class.from_json(params_path)
  if override_hparams is not None:
      hparams.__dict__.update(override_hparams)
  important_hparam_names = override_hparams.keys() if override_hparams is not None else ['layers']
  important_hparams = {k:v for k,v in hparams.__dict__.items() if any([k==name for name in important_hparam_names])}
  exp_name = ROME_experiment_name(args,
                                  model_name.split('/')[-1],
                                  alg_name,
                                  ds_name,
                                  important_hparams)
  return exp_name

def make_editing_results_df(exp_name, n=1000):
  run_dir = os.path.join(f'{BASE_DIR}/results/', exp_name)
  dataframes = []
  printed = 0
  for case_id in range(n):
    case_result_path = os.path.join(run_dir, f"case_{case_id}.json")
    if not os.path.exists(case_result_path):
      if printed < 10:
        print("skipping ", case_result_path, " does not exist")
        printed+=1
      continue
    with open(case_result_path, 'r') as f:
      record = json.load(f)
  
    rewrite_data = record['requested_rewrite']
    prompt = rewrite_data['prompt'].format(rewrite_data['subject'])
    target = rewrite_data['target_true']['str']
    try:
        record_dict = {
            'case_id': [record['case_id']],
            'prompt': [prompt],
            'target': [target],
            'subject' : [rewrite_data['subject']],
            'request' : [rewrite_data['target_new']['str']],
            'request_baseline': [rewrite_data['request_baseline']],
            'tgt_in_sample': [record['tgt_in_sample']],
            # 'tgt_in_sample_2': [record['tgt_in_sample_2']],
            'attack_frac': [record['attack_frac']],
            'retain_rate': [record['retain_rate']],
            'retain_rate_neighborhood': [record['retain_rate_neighborhood']],
            'retain_rate_pre': [record['retain_rate_pre']],
            'retain_rate_neighborhood_pre': [record['retain_rate_neighborhood_pre']],
            'delta_accuracy': [record['delta_accuracy']],
            'delta_accuracy_neighborhood': [record['delta_accuracy_neighborhood']],
            'actual_retain_rate': [record['actual_retain_rate']],
            'actual_retain_rate_neighborhood': [record['actual_retain_rate_neighborhood']],
        }
    except:
        print("skipping ", case_result_path, " missing basic info")
        continue
    cur_sum = collections.defaultdict(lambda: [])
    data = record
    # record difference in pre and post probs for target_new
    for data_type in ['rewrite', 'paraphrase', 'neighborhood']:
        post_prob = np.exp(-data['post'][f'{data_type}_prompts_probs'][0]['target_new'])
        pre_prob = np.exp(-data['pre'][f'{data_type}_prompts_probs'][0]['target_new'])
        cur_sum[f'{data_type}_prob_diff'] = post_prob - pre_prob
        cur_sum[f'{data_type}_pre_prob'] = pre_prob
        cur_sum[f'{data_type}_post_prob'] = post_prob
        erased_prop = (pre_prob - post_prob) / pre_prob
        erased_prop = np.max([0, erased_prop])
        recovered_prop = 1 - (1 - post_prob) / (1 - pre_prob)
        recovered_prop = np.max([2e-8, recovered_prop])
        abs_diff = np.abs(post_prob-pre_prob)
        cur_sum[f'{data_type}_recovered'] = recovered_prop
        cur_sum[f'{data_type}_erased'] = erased_prop
        max_abs_diff = np.abs(pre_prob - .5) + .5
        if data_type != 'neighborhood':
            cur_sum[f'{data_type}_score'] = erased_prop if args.fact_erasure else recovered_prop
        else:
            cur_sum[f'{data_type}_score'] = 1 - abs_diff / max_abs_diff
    try:
        target_score = hmean([
            cur_sum['rewrite_score'], cur_sum['paraphrase_score'], cur_sum['neighborhood_score']
        ])
    except:
        target_score = 0
    cur_sum["target_score"] = target_score
    # compute essence scores 
    if 'essence_score' in data["post"]:
        cur_sum[f"post_essence_ppl"] = data["post"]['essence_score']
        cur_sum[f"pre_essence_ppl"] = data["pre"]['essence_score']
        cur_sum['essence_ppl_diff'] = cur_sum['post_essence_ppl'] - cur_sum['pre_essence_ppl'] # lower is better
    # compute original ROME metrics
    for prefix in ["pre", "post"]:
        # Probability metrics for which new should be lower (better) than true
        for key in ["rewrite_prompts_probs", "paraphrase_prompts_probs"]:
            if prefix not in data or key not in data[prefix]:
                continue
            sum_key_discrete = f"{prefix}_{key.split('_')[0]}_success"
            sum_key_cont = f"{prefix}_{key.split('_')[0]}_diff"
            cur_sum[sum_key_discrete].append(
                np.mean(
                    [
                        x["request_baseline"] > x["target_new"]
                        for x in data[prefix][key]
                    ]
                )
            )
            cur_sum[sum_key_cont].append(
                np.mean(
                    [
                        np.exp(-x["target_new"]) - np.exp(-x["request_baseline"])
                        for x in data[prefix][key]
                    ]
                )
            )
        # Probability metrics for which true should be lower (better) than new
        sum_key_discrete = f"{prefix}_neighborhood_success"
        sum_key_cont = f"{prefix}_neighborhood_diff"
        key = "neighborhood_prompts_probs"
        if prefix in data and key in data[prefix]:
            cur_sum[sum_key_discrete].append(
                np.mean(
                    [
                        x["request_baseline"] < x["target_new"]
                        for x in data[prefix][key]
                    ]
                )
            )
            cur_sum[sum_key_cont].append(
                np.mean(
                    [
                        np.exp(-x["request_baseline"]) - np.exp(-x["target_new"])
                        for x in data[prefix][key]
                    ]
                )
            )
        # zsRE evaluation metrics
        for key in ["rewrite", "paraphrase", "neighborhood"]:
            sum_key = f"{prefix}_{key}_acc"
            key = f"{key}_prompts_correct"
            if prefix not in data or key not in data[prefix]:
                continue
            cur_sum[sum_key].append(np.mean(data[prefix][key]))
        # get harmonic mean averages per point
        for prefix in ["pre", "post"]:
            for k_efficacy, k_generalization, k_specificity in [(
                    f"{prefix}_rewrite_success",
                    f"{prefix}_paraphrase_success",
                    f"{prefix}_neighborhood_success",
                ),
                (
                    f"{prefix}_rewrite_acc",
                    f"{prefix}_paraphrase_acc",
                    f"{prefix}_neighborhood_acc",
                )]:
                  if k_generalization in cur_sum and k_specificity in cur_sum:
                      cur_sum[f"{prefix}_score"] = hmean([
                                  cur_sum[k_efficacy][0],
                                  cur_sum[k_generalization][0],
                                  cur_sum[k_specificity][0]]
                      )
    # add ROME metrics to record_dict and append to dataframes
    record_dict.update(cur_sum)
    df = pd.DataFrame(record_dict)
    dataframes.append(df)
  if len(dataframes) > 0:
    return_df = pd.concat(dataframes)
  else:
    return_df = pd.DataFrame()
  return return_df


def get_subject_noising_function(model, e_range, hparams, embed_layername):
    # define noise embeddings function
    prng = np.random.RandomState(1) 
    # define function that noises embeddings at tokens_to_mix indices
    def noise_embeddings_f(x, layer):
        # skip noising if seq is a single token (must be bos/eos for open-ended generation)
        if (x.shape[1] == 1):
            return x
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if e_range is not None:
                b, e = e_range
                embeds_noise = torch.from_numpy(prng.randn(x.shape[0], e - b, x.shape[2])).to(x.device)
                x[:, b:e] += hparams.editing_noise * embeds_noise
            # print("added noise to embeds: ", embeds_noise)
            return x
        else:
            return x
    return noise_embeddings_f

def main(
    args,
    alg_name: str,
    model_name: Union[str, Tuple],
    ds_name: str,
    dataset_size_limit: int,
    do_essence_tests: bool,
    skip_generation_tests: bool,
    conserve_memory: bool,
    mt=None,
    verbose=False,
    override_hparams=None,
    overwrite=False,
    correctness_check=False,
    target_prob_check=0,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # isin_top5_sum = torch.zeros(49)
    # target_probs_sum = torch.zeros(49)
    # isin_top5_sum_preedit = torch.zeros(49)
    # target_probs_sum_preedit = torch.zeros(49)
    # total_samp_exec = 0
    # Get run hyperparameters
    _model_name = model_name.replace('/', '_')
    params_path = os.path.join(f'{CODE_DIR}/hparams/', alg_name, f"{_model_name}.json")
    if alg_name == 'FT':
      params_path = params_path.replace('.json', '_constr.json')
    hparams = params_class.from_json(params_path)
    args.hparams = hparams
    if override_hparams is not None:
      hparams.__dict__.update(override_hparams)
    print(f"Executing {alg_name} with parameters {hparams}")

    # Determine run directory
    important_hparam_names = override_hparams.keys() if override_hparams is not None else ['layers']
    important_hparams = {k:v for k,v in hparams.__dict__.items() if any([k==name for name in important_hparam_names])}
    exp_name = ROME_experiment_name(args,
                                    model_name.split('/')[-1],
                                    alg_name,
                                    ds_name,
                                    important_hparams)
    run_dir = f'{BASE_DIR}/results/{exp_name}'
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results will be stored at {run_dir}")
    # copy hparams to results dir
    copy_to_path =  os.path.join(run_dir, 'hparams.json')
    if not os.path.exists(copy_to_path):
        shutil.copyfile(params_path, copy_to_path)
    
    # Instantiate vanilla model
    if mt is None:
      print("Instantiating model")
      model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
      tok = AutoTokenizer.from_pretrained(model_name)
      tok.pad_token = tok.eos_token
    else:
      model, tok = mt.model, mt.tokenizer
      tok.pad_token = tok.eos_token

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR)
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None
    print(DATA_DIR)    

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, size=dataset_size_limit, tok=tok)
    # Iterate through dataset
    num_datapoints_executed = 0
    for record in tqdm(ds):
        if (num_datapoints_executed>=args.datapoints_to_execute):
            break
        case_id = record["case_id"] if 'case_id' in record else 'known_id'
        if args.second_half and case_id < 10000:
            continue
        # if case_id < 530:
        #     continue
        # if case_id<198:
        #     continue
        case_result_path = os.path.join(run_dir, f"case_{case_id}.json")
        rewrite_this_point = overwrite or not os.path.exists(case_result_path)
         # skip some weird memory issues
        if case_id == 1531:
            continue
        if case_id == 1517 and ((args.alg_name == "ROME" and args.tracing_reversal) or len(hparams.layers) > 1):
            continue
        if rewrite_this_point:
            print("Starting point: ", case_id)
            # print info for this point
            if args.dummy_string:
                record["requested_rewrite"]['target_new']['str']="dummy"
            request = record["requested_rewrite"]
            subject = record["requested_rewrite"]['subject']
            prompt = request['prompt'].format(subject)
            request['full_prompt'] = prompt
            target_true = request['target_true']['str']
            paraphrase_prompts = record["paraphrase_prompts"]
            print("Evaluation parap")
            print(paraphrase_prompts)
            neighborhood_prompts = record["neighborhood_prompts"]
            if verbose:
                print("Updating point:"
                    f" orig update: [{prompt}] -> [{request['target_new']['str']}]"
                    f"\n True label: {target_true}"
                    f"\n Paraphrases: {paraphrase_prompts[:2]}"
                    f"\n Neighbors: {neighborhood_prompts[:2]}")
            if args.target_tokens_check:
                if len(tok.encode(" "+request['target_new']['str']))!=1:
                    continue
            # check if we should skip based on correctness and probability checks
            if correctness_check or target_prob_check > 0:
                is_correct, meets_target_prob = True, True
                if correctness_check:
                    # print(prompt)
                    gen_batch = simple_make_inputs(tok, prompts=[prompt])
                    samples, scores, _ = predict_model(mt, 
                                            [prompt], 
                                            answers=None, 
                                            trigger_phrase=None, 
                                            max_decode_steps=36)
                 
                    is_correct = fewshot_accuracy_sum(samples, [target_true])
                    

                # sampled_prompts, sampled_targets = [x['requested_rewrite']['prompt'].format(x['requested_rewrite']['subject']) for x in sampled], [x['requested_rewrite']['target_true']['str'] for x in sampled]

                # # print("bf")
                # samples = [predict_model(mt, [sampled_prompts[i]], answers=None, trigger_phrase=None, 
                #                 max_decode_steps=36)[0][0] for i in range(len(sampled_prompts))]
                # # print("af")
                # print(samples)
                # print(sampled_prompts)
                # print(sampled_targets)
                # is_correct = fewshot_accuracy_sum(samples, sampled_targets)
                # print(is_correct)
                # exit()
                    # # print(target_prob_check)
                if target_prob_check > 0:
                    # print("x")
                    preds, scores, _ = predict_model(mt, [prompt], answers=[target_true])
                    # print("x")
                    meets_target_prob = scores[0].item() > target_prob_check
                    # print(meets_target_prob)
                if not (is_correct and meets_target_prob):
                    if verbose:
                        print(" Skipping this point due to it being incorrect or not meeting the minimum target prob.")
                        if target_prob_check > 0: 
                            print(f" Target prob: {scores[0].item():.4f}")
                            print(f" Pred: {preds}")
                    continue

            # generate essence_texts for evaluation if needed
            if do_essence_tests or not skip_generation_tests:
                essence_prompt = "{} is a".format(subject)
                if verbose:
                    print("GENERATING ESSENCE TEXTS")
                # print(essence_prompt)
                essence_texts = generate_fast(
                    model,
                    tok,
                    [essence_prompt],
                    n_gen_per_prompt=5,
                    max_out_len=100,
                )
                # print(essence_texts)
                snips.names_to_samples[subject] = essence_texts
                if verbose:
                    for text in snips.names_to_samples[request['subject']][:2]:
                        print(f" Essence text: {text[:200]}")

            # adjust targets and define 'request_baseline' based on objectives. note model does not necesarily predict 'request_baseline' value before rewriting
            num_noise_samples = 10 if args.fact_forcing and args.alg_name == "FT" else 1 # does not do anything with ROME / MEMIT since these are very low variance
            e_range = find_token_range(tok, substring=subject, prompt_str=prompt)
            request['e_range'] = e_range
            prior_prob = None
            # make noise embeddings_f
            embed_layername = layername(model, 0, 'embed')
            noise_embeddings_f = get_subject_noising_function(model, e_range, hparams, embed_layername)
            num_datapoints_executed += 1
            if args.tracing_reversal:
                gen_batch = simple_make_inputs(tok, prompts=[prompt] * (num_noise_samples))
                with torch.no_grad(), nethook.TraceDict(model, [embed_layername], edit_output=noise_embeddings_f) as td:
                    essence_texts = generate_fast(
                        model,
                        tok,
                        [prompt],
                        n_gen_per_prompt=1,
                        max_out_len=12,
                    )
                    new_target = essence_texts[0]
                request['request_baseline'] = request['target_true']['str']
                request['target_new']['str'] = new_target
                request['target_new']['id'] = 'noised-input'
                if verbose:
                    score_batch = make_inputs(tok, [prompt], targets=[new_target])
                    init_target_prob = score_from_batch(model, score_batch)
                    print(f" NEW TARGET PREDICTION: {new_target}")
                    print(f" with init pred prob: {init_target_prob.item():.4f}")
            elif args.fact_erasure:
                batch = make_inputs(mt.tokenizer, prompts=[prompt] * num_noise_samples, targets=[target_true] * num_noise_samples)
                request['request_baseline'] = mt.tokenizer.eos_token # arbitrary token, won't use these metrics anyway
                request['target_new'] = request['target_true']
                print(request['target_new'])
            elif args.fact_amplification:
                request['request_baseline'] = mt.tokenizer.eos_token # arbitrary token, won't use these metrics anyway
                request['target_new'] = request['target_true']
            elif args.fact_forcing or args.weight_based_tracing:
                gen_batch = simple_make_inputs(tok, prompts=[prompt] * (num_noise_samples))
                _, noised_pred_id = corrupted_forward_pass(mt.model, None, gen_batch, tokens_to_mix=e_range, noise=hparams.editing_noise)
                noised_pred_token = tok.decode([noised_pred_id])
                request['request_baseline'] = noised_pred_token
                request['target_new'] = request['target_true']
            else:
                request['request_baseline'] = request['target_true']['str']
            if verbose:
                print(" request baseline: ", request['request_baseline'])
                
            # get additional functions and variables based on objectives
            # get hidden representations from corrupted+uncorrupted forward passes to use as targets for weight editing
            if args.weight_based_tracing:
                last_subj_idx = e_range[1]
                with torch.enable_grad():
                    # corrupted forward pass. corrupted_hidden_states will be of shape [n_layers, num_noise_samples, seq_len, hidden_dim]
                    gen_batch = simple_make_inputs(tok, prompts=[prompt] * num_noise_samples)
                    gen_batch['output_hidden_states'] = True
                    _, _, corrupted_hidden_states = corrupted_forward_pass(model, None, gen_batch, tokens_to_mix=e_range, noise=hparams.editing_noise, output_hidden_states=True)
                    corrupted_hidden_states = torch.stack([corrupted_hidden_states[layer+1] for layer in hparams.layers], dim=0)
                    # clean forward pass
                    gen_batch = simple_make_inputs(tok, prompts=[prompt])
                    clean_hidden_states = model(**gen_batch, output_hidden_states=True).hidden_states
                    clean_hidden_states = torch.stack([clean_hidden_states[layer+1] for layer in hparams.layers], dim=0)
                # splice uncorrupted hidden_states into corrupted_hidden_states where they are restored. automatically broadcast across num_noise_samples dimension
                hidden_state_supervision = corrupted_hidden_states
                hidden_state_supervision[:,:,last_subj_idx,:] = clean_hidden_states[:,:,last_subj_idx,:]
            else:
                hidden_state_supervision = None

            # Compute weight changes + record weights that changed
            start = time.time()
            args_conserve_memory = (
                dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
                if conserve_memory
                else dict()
            )

            if args.retain_rate:
                sampled = sample(retain_rate_samples, 100)
                query_inputs, sampled_targets = [x['requested_rewrite']['prompt'].format(x['requested_rewrite']['subject']) for x in sampled], [x['requested_rewrite']['target_true']['str'] for x in sampled]
                # print(query_inputs)
                # print(sampled_targets)
                # exit()
                batch = simple_make_inputs(tok, query_inputs)
                pad_token_id = tok.pad_token_id
                outputs = model.generate(**batch, do_sample=False, max_new_tokens=36,
                                  pad_token_id=pad_token_id)
                outputs = [list(filter(lambda x: x != pad_token_id, output)) for output in outputs]
                preds = [tok.decode(output) for output in outputs]
                preds = [pred.replace(query_input, "").strip() for pred, query_input in zip(preds, query_inputs)]
                retain_rate_pre = fewshot_accuracy_sum(preds, sampled_targets)/len(preds)
                preds_preedit = preds
        
                
                query_inputs, sampled_targets = record['neighborhood_prompts'], [record['requested_rewrite']['target_true']['str'] for x in range(len(record['neighborhood_prompts']))]
                
                
                batch = simple_make_inputs(tok, query_inputs)
                outputs = model.generate(**batch, do_sample=False, max_new_tokens=36,
                                  pad_token_id=pad_token_id)
                outputs = [list(filter(lambda x: x != pad_token_id, output)) for output in outputs]
                preds = [tok.decode(output) for output in outputs]
                preds = [pred.replace(query_input, "").strip() for pred, query_input in zip(preds, query_inputs)]
                retain_rate_neighborhood_pre = fewshot_accuracy_sum(preds, sampled_targets)/len(preds)

                preds_preedit_neighborhood = preds

                

            with torch.enable_grad(), nethook.TraceDict(model, [embed_layername], edit_output=noise_embeddings_f) if args.fact_forcing else nullcontext() as td:
              edited_model, weights_copy = apply_algo(
                  args,
                  model,
                  tok,
                  [request],
                  hparams,
                  copy=False,
                  return_orig_weights=True,
                  num_noise_samples=num_noise_samples,
                  prior_prob=prior_prob,
                  hidden_state_supervision=hidden_state_supervision,
                  **args_conserve_memory,
              )
            exec_time = time.time() - start
            print("Execution took", exec_time)
            # exit()
            # state_a = model.state_dict().__str__()
            # state_b = edited_model.state_dict().__str__()
            # if state_a == state_b:
            #     print("same weight")
            # else:
            #     print("diff weight")
            # exit()

            # Execute evaluation suite
            if args.retain_rate:
                
                query_inputs, sampled_targets = [x['requested_rewrite']['prompt'].format(x['requested_rewrite']['subject']) for x in sampled], [x['requested_rewrite']['target_true']['str'] for x in sampled]
                # print(query_inputs)
                # print(sampled_targets)
                # exit()
                batch = simple_make_inputs(tok, query_inputs)
                pad_token_id = tok.pad_token_id
                outputs = edited_model.generate(**batch, do_sample=False, max_new_tokens=36,
                                  pad_token_id=pad_token_id)
                outputs = [list(filter(lambda x: x != pad_token_id, output)) for output in outputs]
                preds = [tok.decode(output) for output in outputs]
                preds = [pred.replace(query_input, "").strip() for pred, query_input in zip(preds, query_inputs)]
                retain_rate = fewshot_accuracy_sum(preds, sampled_targets)/len(preds)
                actual_retain_rate =  fewshot_accuracy_sum(preds, preds_preedit)/len(preds)
                
        
                
                query_inputs, sampled_targets = record['neighborhood_prompts'], [record['requested_rewrite']['target_true']['str'] for x in range(len(record['neighborhood_prompts']))]
                
                
                batch = simple_make_inputs(tok, query_inputs)
                outputs = edited_model.generate(**batch, do_sample=False, max_new_tokens=36,
                                  pad_token_id=pad_token_id)
                outputs = [list(filter(lambda x: x != pad_token_id, output)) for output in outputs]
                preds = [tok.decode(output) for output in outputs]
                preds = [pred.replace(query_input, "").strip() for pred, query_input in zip(preds, query_inputs)]
                retain_rate_neighborhood = fewshot_accuracy_sum(preds, sampled_targets)/len(preds)
                actual_retain_rate_neighborhood = fewshot_accuracy_sum(preds, preds_preedit_neighborhood)/len(preds)
                
                
                
            start = time.time()
            with torch.no_grad():
              if args.attack == "mg":
                    paraps = rephrases[request['prompt'].format(request['subject'])+" "+request['target_true']['str']]
                    paraps = [request['prompt'].format(request['subject'])] + paraps
              else:
                    prompt_cur = request['prompt'].format(request['subject'])
                    paraps = [pref + prompt_cur for pref in prefixes]
            
              if len(paraps)>args.num_attack_parap:
                num_attack_parap = args.num_attack_parap
              else: 
                num_attack_parap = len(paraps)
              print(num_attack_parap)
              print("attack parap")
            #   print(paraps[:num_attack_parap])
              attack_paraps = paraps[-num_attack_parap:]
              attack_paraps = [x.replace(request["target_true"]["str"],"") for x in attack_paraps]
            #   if(len(paraps)>4):
            #     attack_paraps = paraps[4:5]
            #   else:
            #     attack_paraps = paraps[1:2]
              print(attack_paraps)
            #   inp_tok = tok(attack_paraps, padding=True, return_tensors="pt").to(device)
            #   input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
              batch = simple_make_inputs(tok, attack_paraps)
              pad_token_id = tok.pad_token_id
              pad_token_id = pad_token_id if pad_token_id else 0
              np.random.seed(args.seed+2)
              torch.cuda.set_device(device)
              torch.random.manual_seed(args.seed+2)
              torch.cuda.manual_seed_all(args.seed+2)
            #   outputs = edited_model.generate(**batch, do_sample=True, num_beams=1, max_new_tokens=1,
            #                       pad_token_id=pad_token_id, num_return_sequences=1, top_k=3)
              torch.manual_seed(42)
            #   set_seed(42)
              outputs = edited_model.generate(**batch, do_sample=True, max_new_tokens=1,
                                  pad_token_id=pad_token_id, num_return_sequences=args.bb_num_samples)
            #   print(outputs.shape)
            #   exit()

              outputs = [list(filter(lambda x: x != pad_token_id, output)) for output in outputs]
              preds = [tok.decode(output) for output in outputs]
              
              
              preds = [pred.replace(query_input, "").replace("!", "").strip() for pred, query_input in zip(preds, [element for element in attack_paraps for i in range(args.bb_num_samples)])] 
            #   print(preds)
         
              
              preds_attack_succ = [x for x in preds if request['target_true']['str'] in x]
              
            #   preds = preds.view(num_return_sequences, num_attack_parap, -1)
            #   preds = [pred.replace(query_input, "").replace("!", "").strip() for pred, query_input in zip(preds, paraps[:num_attack_parap])]     
            #   # for some reason huggingface generate not giving generation probs, so we recalculate
            #   targets = [request['target_new']['str'] for pred, query_input in zip(preds, paraps[:num_attack_parap])]
            #   print(preds)
            #   print(targets)
            #   batch = make_inputs(tok, paraps[:num_attack_parap], targets=preds)
            #   scores = score_from_batch(edited_model, batch)
            #   print(scores)
            #   batch = make_inputs(tok, paraps[:num_attack_parap], targets=targets)
            #   scores = score_from_batch(edited_model, batch)
            #   print(scores)
            #   exit()
              
            #   print(edited_model(input_ids).logits.shape)
            #   softmax_out = torch.nn.functional.softmax(edited_model(input_ids).logits[:, -1, :], dim=1)
            #   target_ids = torch.as_tensor(tok.encode(request['target_new']['str'])).view(1, -1).to(device)
            #   target_prob = softmax_out[:, target_ids[0][0]]
            #   print(target_prob)
            #   exit()
              
            #   tk = torch.topk(softmax_out, k=50257, dim=1).indices
            #   softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            #   exit()
            #   softmax_out_top_k = softmax_out
            #   softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            #   new_tok_indices = torch.multinomial(softmax_out_top_k, 50, replacement=True)
            #   new_toks = torch.gather(tk, 1, new_tok_indices)
              
            #   print(target_ids[0][0] in new_toks)
            #   print(target_ids[0][0])
            #   print(new_toks)
            #   exit()
            #   print(tok.decode(target_ids[0]))
            #   print([tok.decode(new_toks[i]) for i in range(new_toks.shape[0])])
            #   print(target_ids[0][0] in new_toks)
              print("tgt_in_sample")
              print(len(preds_attack_succ))
            
              metrics = {
                    "case_id": case_id,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": ds_eval_method(args, edited_model, tok, record, snips, vec, skip_generation_tests),
                    "tgt_in_sample": len(preds_attack_succ)>0,#(target_ids[0][0] in new_toks),
                    # "tgt_in_sample_2": (target_ids[0][0] in new_toks),
                    "attack_frac": len(preds_attack_succ)/len(preds),
                    "retain_rate": retain_rate,
                    "retain_rate_neighborhood": retain_rate_neighborhood,
                    "actual_retain_rate": actual_retain_rate,
                    "actual_retain_rate_neighborhood": actual_retain_rate_neighborhood,
                    "retain_rate_pre" : retain_rate_pre,
                    "retain_rate_neighborhood_pre" : retain_rate_neighborhood_pre
                }

                

              
              
            #   for i in range(num_attack_parap):
            #    if paraps[i]:
            #     input_ids = torch.as_tensor(tok.encode(paraps[i])).view(1, -1).to(device)
            #     target_ids = torch.as_tensor(tok.encode(" "+request['target_new']['str'])).view(1, -1).to(device)
            #     # input_tok = tok([prompt.format(request["subject"]) for prompt in all_prompts], return_tensors="pt", padding=True,).to("cuda")
            #     # target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")["input_ids"][0]
            #     top_k, bottom_k, top_1, bottom_1, target_prob = get_metrics(edited_model, input_ids, target_ids)
            #     metrics = {
            #         "case_id": str(case_id)+'_p_'+str(i),
            #         "requested_rewrite": request,
            #         "time": exec_time,
            #         "post": ds_eval_method(args, edited_model, tok, record, snips, vec, skip_generation_tests),
            #         "topk_metric": list(top_k.to(dtype=torch.float64).cpu().numpy()),
            #         "bottomk_metric": list(bottom_k.to(dtype=torch.float64).cpu().numpy()),
            #         "top1_metric": list(top_1.to(dtype=torch.float64).cpu().numpy()),
            #         "bottom1_metric": list(bottom_1.to(dtype=torch.float64).cpu().numpy()),
            #         "target_prob": list(target_prob.to(dtype=torch.float64).cpu().numpy())
            #     }
                # tok.encode(text)
                # input_ids = torch.as_tensor(tok.encode(request['prompt'].format(request['subject']))).view(1, -1).to(device)
                # print(request['prompt'].format(request['subject']))
                # target_ids = torch.as_tensor(tok.encode(" "+request['target_new']['str'])).view(1, -1).to(device)
                # print(target_ids)
                # print(len(request['target_new']['str']))
                # total_samp_exec += 1
                # isin_top5_sum += isin_top5[:,-1].astype(int)
                # target_probs_sum += probs[:,-1]
                
                
                # exit()
                # print(edited_model.state_dict()['transformer.h.17.mlp.c_proj.weight'])
              for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to("cuda")
              metrics["pre"] = ds_eval_method(args, model, tok, record, snips, vec, skip_generation_tests)
              metrics['prior_prob'] = prior_prob
              if args.retain_rate:
                # query_inputs, sampled_targets = [x['requested_rewrite']['prompt'].format(x['requested_rewrite']['subject']) for x in sampled], [x['requested_rewrite']['target_true']['str'] for x in sampled]
                # # print(query_inputs)
                # # print(sampled_targets)
                # # exit()
                # batch = simple_make_inputs(tok, query_inputs)
                # pad_token_id = tok.pad_token_id
                # outputs = edited_model.generate(**batch, do_sample=False, max_new_tokens=36,
                #                   pad_token_id=pad_token_id)
                # outputs = [list(filter(lambda x: x != pad_token_id, output)) for output in outputs]
                # preds = [tok.decode(output) for output in outputs]
                # preds = [pred.replace(query_input, "").strip() for pred, query_input in zip(preds, query_inputs)]
                # retain_rate_pre = fewshot_accuracy_sum(preds, sampled_targets)/len(preds)
                
        
                
                # query_inputs, sampled_targets = record['neighborhood_prompts'], [record['requested_rewrite']['target_true']['str'] for x in range(len(record['neighborhood_prompts']))]
                
                
                # batch = simple_make_inputs(tok, query_inputs)
                # outputs = edited_model.generate(**batch, do_sample=False, max_new_tokens=36,
                #                   pad_token_id=pad_token_id)
                # outputs = [list(filter(lambda x: x != pad_token_id, output)) for output in outputs]
                # preds = [tok.decode(output) for output in outputs]
                # preds = [pred.replace(query_input, "").strip() for pred, query_input in zip(preds, query_inputs)]
                # retain_rate_neighborhood_pre = fewshot_accuracy_sum(preds, sampled_targets)/len(preds)

                # metrics["retain_rate_pre"] = retain_rate_pre
                # metrics["retain_rate_neighborhood_pre"] = retain_rate_neighborhood_pre
                metrics["delta_accuracy"] = metrics["retain_rate_pre"]-metrics["retain_rate"]
                metrics["delta_accuracy_neighborhood"] = metrics["retain_rate_neighborhood_pre"]-metrics["retain_rate_neighborhood"]

            
               

            print("Evaluation took", time.time() - start)
            # print(metrics)
            # exit()
            # print(metrics["pre"]['paraphrase_prompts_probs'])
            # print(metrics["post"]['paraphrase_prompts_probs'])
            # exit()
            # Dump metrics in .json
            with open(case_result_path, "w") as f:
                json.dump(metrics, f, indent=1)
            print('\n')
            
        else:
            if verbose:
              print(f"skipping {case_result_path}, already run")
            else:
              pass
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dummy_string",
        action="store_true",
        help="dummy_string",
    )
    parser.add_argument(
        "--alg_name",
        choices=["ROME", "FT", "KN", "MEND", "KE", "MEMIT"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--fact_token",
        choices=["subject_last", "last"],
        default="subject_last",
        help="When using ROME, this is the token index for which to optimize the MLP output vector. last is last in sequence, subject_last is last in subject entity",
        required=False,
    )
    parser.add_argument(
        "--edit_layer",
        type=int,
        default=0,
        help="Layer at which to edit the model weights. Set to -2 to defer to hparam sweep params below",
    )
    parser.add_argument(
        "--ds_name",
        choices=["cf", "cf_filt", "cf_filt_para", "cf_attack", "zsre"],
        default="cf",
        help="Dataset to perform evaluations on. Either CounterFact (cf) or zsRE (zsre).",
    )
    parser.add_argument(
        "--attack",
        choices=["cf", "mg"],
        default="cf",
        help="Type of attack paraphrases",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--window_sizes",
        type=str,
        default='1',
        help="Window sizes separated by spaces to use for editing method",
    )
    parser.add_argument(
        "--dataset_size_limit",
        "-n",
        type=int,
        default=2000,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--datapoints_to_execute",
        "-datapoints_to_execute",
        type=int,
        default=700,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--norm_constraint",
        type=float,
        default=-1,
        help="l0 norm constraint on constrained finetuning",
    )
    parser.add_argument(
        "--kl_factor",
        type=float,
        default=-1,
        help="weight on kl div for essence drift regularization",
    )
    parser.add_argument(
        "--v_lr",
        type=float,
        default=-1,
        help="learning rate for finding v* vector in ROME",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previous experiment results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="More printing during editing",
    )
    parser.add_argument(
        "--target_tokens_check",
        action="store_true",
        help="target tokens=1",
    )
    parser.add_argument(
        "--second_half",
        action="store_true",
        help="iterate over second_half of cf",
    )
    parser.add_argument(
        "--retain_rate",
        action="store_true",
        help="compute retain rate",
    )
    parser.add_argument(
        "--do_essence_tests",
        type=int,
        default=1,
        choices=[0,1],
        help="Do the essence drift generation test regardless of args.skip_generation_tests",
    )
    parser.add_argument(
        "--k",
        "-k",
        type=int,
        default=4,
        help="top-k metric",
    )
    parser.add_argument(
        "--bb_num_samples",
        "-bb_num_samples",
        type=int,
        default=4,
        help="--bb_num_samples",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--tracing_reversal",
        action="store_true",
        help="Rather than changing output from target_true to target_new, change it to the prediction obtained from the noised causal tracing input",
    )
    parser.add_argument(
        "--fact_forcing",
        action="store_true",
        help="Rather than change o-true to o-new for (s,r,.) input, change o-noise to o-true for (s-noise, r,.) input",
    )
    parser.add_argument(
        "--fact_erasure",
        action="store_true",
        help="See paper for description",
    )
    parser.add_argument(
        "--fact_amplification",
        action="store_true",
        help="See paper for description",
    )
    parser.add_argument(
        "--weight_based_tracing",
        action="store_true",
        help="See paper for description",
    )
    parser.add_argument(
        "--correctness_filter",
        type=int,
        default=0,
        choices=[0,1],
        help="Only eval on points with correct generations or p(target_true) >= .1",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--margin_loss",
        dest="margin_loss",
        action="store_true",
        help="Add margin loss",
    )
    parser.add_argument(
        "--entropy_loss",
        dest="entropy_loss",
        action="store_true",
        help="Add entropy loss",
    )
    parser.add_argument(
        "--grad_margin_loss",
        dest="grad_margin_loss",
        action="store_true",
        help="Add grad margin loss turn --margin_loss true while using this",
    )
    parser.add_argument(
        "--analytical_soln",
        dest="analytical_soln",
        action="store_true",
        help="replace v* by negative of the embedding of the target token",
    )

    parser.add_argument(
        "--model_parap",
        dest="model_parap",
        action="store_true",
        help="use model paraphrases",
    )

    parser.add_argument(
        "--cf_defense",
        dest="cf_defense",
        action="store_true",
        help="use model paraphrases",
    )
    
    parser.add_argument(
        "--margin_layers",
        dest="margin_layers",
        nargs='+', type=int, help='layers on which margin loss is to be applied', required=False
    )
    parser.add_argument(
        "--entropy_layers",
        dest="entropy_layers",
        nargs='+', type=int, help='layers on which margin loss is to be applied', required=False
    )
    parser.add_argument(
        "--run",
        type=int,
        default=1,
        choices=[0,1],
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_attack_parap",
        type=int,
        default=5,
    )
    

    parser.set_defaults(skip_generation_tests=True, conserve_memory=True)
    args = parser.parse_args()

    # set device
    device = torch.device(f"cuda:{args.gpu}")
    np.random.seed(args.seed)
    torch.cuda.set_device(device)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # experiment checks
    if args.fact_erasure:
        assert args.correctness_filter, "only erase known facts"
    # if args.alg_name == "MEMIT":
    #     assert args.window_sizes != "1", "use window size >=1 with MEMIT"

    # load model
    if args.run:
        torch.set_grad_enabled(False)
        model_name = args.model_name
        torch_dtype = torch.float16 if '20b' in model_name else None
        mem_usage = True
        print("Loading model...")
        if '20b' not in model_name:
            mt = ModelAndTokenizer(model_name, low_cpu_mem_usage=mem_usage, torch_dtype=torch_dtype, cache_dir=MODEL_DIR)
            torch.cuda.empty_cache()
            mt.model.eval().cuda()
            mt.tokenizer.add_special_tokens({'pad_token' : mt.tokenizer.eos_token})
        else:
            raise RuntimeError("20b model does not load properly across devices")
            from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
            model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", 
                                                        device_map={
                                                            'embed_out' : 0,
                                                            'gpt_neox.embed_in' : 0,
                                                            'gpt_neox.layers': 1,
                                                            'gpt_neox.final_layer_norm' : 0,
                                                        },
                                                        low_cpu_mem_usage=mem_usage,
                                                        torch_dtype=torch_dtype)
            torch.cuda.empty_cache()
            model.eval().cuda()
            tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
            mt = ModelAndTokenizer(model=model, tokenizer=tokenizer, torch_dtype=torch_dtype)
            mt.tokenizer.add_special_tokens({'pad_token' : mt.tokenizer.eos_token})

    # print(mt.model)
    # word_embeddings = mt.model.transformer.wte.weight
    # lm_head = mt.model.lm_head.weight
    # print(word_embeddings.shape)
    # print(lm_head.shape)
    # print(word_embeddings == lm_head)
    # # print(word_embeddings.shape)
    # exit()

    # set experiment args
    RUN_EXPERIMENT = args.run # set to false to just collect results
    num_points = args.dataset_size_limit
    alg_name = args.alg_name
    model_name = args.model_name
    assert alg_name in ["FT", "ROME", "MEMIT"]
    hparams_class, _ = ALG_DICT[alg_name]
    ds_name = args.ds_name
    window_sizes = [int(x) for x in args.window_sizes.split()]
    if 'gpt2' in model_name:
        central_layers = list(range(0, 48, 4)) + [17, 47]
        num_layers = 48
    if '6B' in model_name:
        central_layers = list(range(0, 28, 4)) + [5, 27]
        num_layers = 28
    if alg_name == 'FT' and 1 in window_sizes and not args.fact_forcing:
        central_layers = [-1] + central_layers
    if args.edit_layer > -2:
        central_layers = [args.edit_layer]
    print("Starting sweep with hparams:")
    print("- window_sizes: ", window_sizes)
    print("- central_layers: ", central_layers)

    # main experiment loop
    results_dfs = []
    for window_size in window_sizes:
        for central_layer in central_layers:
            override_hparams = get_override_hparams(args, window_size, central_layer, alg_name)
            if RUN_EXPERIMENT:
                main(
                    args,
                    alg_name=alg_name,
                    model_name=model_name,
                    ds_name=ds_name,
                    dataset_size_limit=num_points,
                    do_essence_tests=args.do_essence_tests,
                    skip_generation_tests=args.skip_generation_tests,
                    conserve_memory=args.conserve_memory,
                    mt=mt,
                    override_hparams=override_hparams,
                    verbose=args.verbose,
                    overwrite=args.overwrite,
                    correctness_check=args.correctness_filter,
                    target_prob_check=.02 if args.correctness_filter and args.fact_erasure else 0
                )
            # accumulate results
            exp_name = ROME_experiment_name_from_override_params(args, model_name, alg_name, ds_name, override_hparams, hparams_class)
            editing_results_df = make_editing_results_df(exp_name, n=num_points)
            editing_results_df['edit_method'] = alg_name
            editing_results_df['edit_central_layer'] = central_layer
            editing_results_df['edit_window_size'] = window_size
            results_dfs.append(editing_results_df)
    
    # combine and save results
    results_df = pd.concat(results_dfs)
    _model_name = model_name.split('/')[-1]
    sweep_params = {'ws': window_sizes, 'layers': args.edit_layer}
    ovr_exp_name = sweep_experiment_name(args, _model_name, alg_name, ds_name, sweep_params)
    file_name = f'{ovr_exp_name}.csv'
    save_path = f'{BASE_DIR}/results/{file_name}'
    results_df.to_csv(save_path, index=False)

    print(f"saving csv at {save_path}...")
    print("results shape: ", results_df.shape)
 
    metrics = ['actual_retain_rate', 'actual_retain_rate_neighborhood', 'delta_accuracy', 'delta_accuracy_neighborhood', 'retain_rate_neighborhood_pre', 'retain_rate_pre', 'retain_rate_neighborhood', 'retain_rate', 'attack_frac', 'tgt_in_sample', 'post_rewrite_success', 'rewrite_prob_diff', 'rewrite_post_prob', 'rewrite_score', 'post_paraphrase_success', 'paraphrase_prob_diff', 'paraphrase_pre_prob', 'paraphrase_post_prob', 'paraphrase_score', 'post_neighborhood_success', 'neighborhood_prob_diff', 'neighborhood_score', 'essence_ppl_diff']
    # another test code review comment
    if True: #len(window_sizes) == 1 and len(central_layers) == 1:
        print("\nfinal metrics: ")
        avg_metrics = {}
        for metric in metrics:
            if metric in results_df.columns:
                # if metric == 'tgt_in_sample':
                #   print(results_df.loc[:,metric])
                avg_val = np.mean(results_df.loc[:,metric])
                print(f" {metric:.20s}: {avg_val:.3f}")
                avg_metrics[metric] = avg_val
            else:
                print(f" missing {metric}")
    
    
    avg_metrics['name'] = file_name

    
    with open('../results/results_bb.csv', 'a', newline='') as f_object:
            writer = csv.writer(f_object)
            
            # dict = avg_metrics
            writer.writerow(list(avg_metrics.values()))
            # f_object.close()
 
    
