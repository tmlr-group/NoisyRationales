# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import math
import logging
import numpy as np
from typing import List, Any, Dict, Union, Tuple
from tqdm import tqdm

from args import ClassifierArgs
from utils.mask import mask_instance, mask_forbidden_index
from utils.utils import build_forbidden_mask_words
from utils.certify import predict, lc_bound, population_radius_for_majority, population_radius_for_majority_by_estimating_lambda, population_lambda
# from torch.optim.adamw import AdamW


from old_code.denoiser import denoise_instance
import os
import random
from collections import defaultdict
import numpy as np
import logging


class Classifier:
    def __init__(self, args: ClassifierArgs):
        # check mode
        self.methods = {'train': self.train, 
                        'evaluate': self.evaluate,
                        'predict': self.predict, 
                        'attack': self.attack,
                        'augmentation': self.augmentation,
                        'certify': self.certify,
                        'statistics': self.statistics
                        }# 'certify': self.certify}
        assert args.mode in self.methods, 'mode {} not found'.format(args.mode)

        # for data_reader and processing
        self.data_reader, self.tokenizer, self.data_processor = self.build_data_processor(args)
        self.model = self.build_model(args)
        self.type_accept_instance_as_input = ['conat', 'sparse', 'safer']
        self.loss_function = self.build_criterion(args.dataset_name)
        
        self.forbidden_words = None
        if args.keep_sentiment_word:
            self.forbidden_words = build_forbidden_mask_words(args.sentiment_path)

    
    def certify(self, args: ClassifierArgs, alpaca=None,**kwargs):

        category = np.array([0,0,0,0,0,0,0,0,0,0])

        cancate_p_list = []
        cancate_label_list = []
        guess_distri = [0,0,0,0,0,0,0,0,0,0]
        guess_distri_ensemble = [0,0,0,0,0,0,0,0,0,0]

        entropy_list = []

        
        if args.predictor == "bert":
            pass
        elif args.predictor == "alpaca_sst2":
            print('alpaca sst2')
            predictor = alpaca
            alpaca.as_sst2()
        elif args.predictor == "alpaca_agnews":
            print('alpaca agnews')
            predictor = alpaca
            alpaca.as_agnews()
        else:
            raise RuntimeError

        dataset, _ = self.build_data_loader(args, args.evaluation_data_type, tokenizer=False)
        if args.certify_numbers == -1:
            certify_dataset = dataset.data
        else:
            certify_dataset = np.random.choice(dataset.data, size=(args.certify_numbers, ), replace=False)
        print(certify_dataset)

        description = tqdm(certify_dataset)
        num_labels = self.data_reader.NUM_LABELS
        
        index_org_sentence = -1
        for data in description:
            index_org_sentence+=1
            if args.stop_iter != -1:
                if args.stop_iter-1 == index_org_sentence:
                    break

            target = self.data_reader.get_label_to_idx(data.label)
            data_length = data.length()
            

            # save or load org sentence
            if data.text_b is None:
                org_sentence_path = os.path.join(args.save_path,"org_sentence",f'{index_org_sentence}-a')
                if args.recover_past_data:
                    if os.path.exists(org_sentence_path):
                        with open(org_sentence_path, 'r') as file:
                            content = file.read()
                            data.text_a = content
                with open(org_sentence_path, 'w') as file:
                    file.write(data.text_a)
            else:
                raise RuntimeError
        
            
            keep_nums = data_length - round(data_length * args.sparse_mask_rate)

            if args.random_probs_strategy != 'None':
                random_probs = alpaca.cal_importance(data,strategy=args.random_probs_strategy)
            else:
                random_probs = None

            tmp_instances = self.mask_instance_decorator(args, data, args.predict_ensemble, random_probs = random_probs)

            for instance in tmp_instances:
                instance.text_a = instance.text_a.replace("<mask>", args.mask_word)
                if instance.text_b is not None:
                    instance.text_b = instance.text_b.replace("<mask>", args.mask_word)

            # save or load pred_masked_sentence
            index_pred_masked_sentence=-1
            if not os.path.exists(os.path.join(args.save_path,"pred_masked_sentence",f"{index_org_sentence}")):
                os.makedirs(os.path.join(args.save_path,"pred_masked_sentence",f"{index_org_sentence}"))
            for instance in tmp_instances:
                index_pred_masked_sentence+=1
                if instance.text_b is None:
                    pred_masked_sentence_path = os.path.join(args.save_path,"pred_masked_sentence",f"{index_org_sentence}",f"a-{index_pred_masked_sentence}")
                    if args.recover_past_data:
                        if os.path.exists(pred_masked_sentence_path):
                            with open(pred_masked_sentence_path, 'r') as file:
                                content = file.read()
                                instance.text_a = content
                    with open(pred_masked_sentence_path, 'w') as file:
                        file.write(instance.text_a)
                else:
                    raise RuntimeError

            if args.denoise_method == None:
                pass
            elif "chatgpt" in args.denoise_method:
                if not os.path.exists(os.path.join(args.save_path,"pred_denoised_sentence",f"{index_org_sentence}",f"a-0")):
                    denoise_instance(tmp_instances, args)
            elif args.denoise_method == 'alpaca':
                if not os.path.exists(os.path.join(args.save_path,"pred_denoised_sentence",f"{index_org_sentence}",f"a-0")):
                    alpaca.denoise_instances(tmp_instances)
            elif args.denoise_method == 'roberta':
                if not os.path.exists(os.path.join(args.save_path,"pred_denoised_sentence",f"{index_org_sentence}",f"a-0")):
                    alpaca.roberta_denoise_instances(tmp_instances)
            elif args.denoise_method == 'remove_mask':
                if not os.path.exists(os.path.join(args.save_path,"pred_denoised_sentence",f"{index_org_sentence}",f"a-0")):
                    for instance in tmp_instances:
                        instance.text_a = instance.text_a.replace(f"{args.mask_word} ", '').replace(f" {args.mask_word}", '')
                        if instance.text_b is not None:
                            instance.text_b = instance.text_b.replace(f"{args.mask_word} ", '').replace(f" {args.mask_word}", '')


            # save or load pred_denoised_sentence
            if args.denoise_method is not None:
                if not os.path.exists(os.path.join(args.save_path,"pred_denoised_sentence",f"{index_org_sentence}")):
                    os.makedirs(os.path.join(args.save_path,"pred_denoised_sentence",f"{index_org_sentence}"))
                index_pred_denoised_sentence=-1
                for instance in tmp_instances:
                    index_pred_denoised_sentence+=1
                    if instance.text_b is None:
                        pred_denoised_sentence_path = os.path.join(args.save_path,"pred_denoised_sentence",f"{index_org_sentence}",f"a-{index_pred_denoised_sentence}")

                        if args.recover_past_data:
                            if os.path.exists(pred_denoised_sentence_path):
                                with open(pred_denoised_sentence_path, 'r') as file:
                                    content = file.read()
                                    instance.text_a = content
                        with open(pred_denoised_sentence_path, 'w') as file:
                            file.write(instance.text_a)
                    else:
                        raise RuntimeError
                    
            # load pred_prediction
            if args.recover_past_data:
                if os.path.exists(os.path.join(args.save_path,"pred_prediction",f"{index_org_sentence}",f"0")):
                    past_pred_predictions = []
                    for i in range(len(tmp_instances)):
                        with open(os.path.join(args.save_path,"pred_prediction",f"{index_org_sentence}",f"{i}"), 'r') as file:
                            content = file.read()
                            past_pred_predictions.append(content)
                else:
                    past_pred_predictions = None
            else:
                past_pred_predictions = None

            # load pred_prediction_prob
            if args.recover_past_data:
                if os.path.exists(os.path.join(args.save_path,"pred_prediction_prob",f"{index_org_sentence}.npy")):
                    past_pred_predictions_prob = np.load(os.path.join(args.save_path,"pred_prediction_prob",f"{index_org_sentence}.npy"))
                else:
                    past_pred_predictions_prob = None
            else:
                past_pred_predictions_prob = None


            if args.predictor == 'bert':
                tmp_probs = predictor.predict_batch(tmp_instances)
            else:
                tmp_probs, pred_predictions = predictor.predict_batch(tmp_instances,past_pred_predictions,past_pred_predictions_prob)

                # save pred_prediction
                if not os.path.exists(os.path.join(args.save_path,"pred_prediction",f"{index_org_sentence}")):
                    os.makedirs(os.path.join(args.save_path,"pred_prediction",f"{index_org_sentence}"))
                if pred_predictions is not None:
                    for i in range(len(pred_predictions)):
                        pred_prediction_path = os.path.join(args.save_path,"pred_prediction",f"{index_org_sentence}",f'{i}')
                        with open(pred_prediction_path, 'w') as file:
                                file.write(pred_predictions[i])

                # save pred_prediction_prob
                np.save(os.path.join(args.save_path,"pred_prediction_prob",f"{index_org_sentence}.npy"), tmp_probs)




            cancate_p_list.append(tmp_probs)
            cancate_label_list.extend( [target for _ in range(len(tmp_probs))] )
                

            guess = np.argmax(tmp_probs, axis=-1).reshape(-1)
            print(list(guess),np.argmax(np.bincount(np.argmax(tmp_probs, axis=-1), minlength=num_labels)),'|',target,file=log_file,flush=True)

            
            
            guess = np.argmax(np.bincount(np.argmax(tmp_probs, axis=-1), minlength=num_labels))

            guess_distri[guess] += 1

            all_guess = np.argmax(tmp_probs, axis=-1)

            for item in all_guess:
                guess_distri_ensemble[item]+=1
            
            
            category[target] += 1


            if guess != target:
                radius = np.nan  

                tmp_instances = self.mask_instance_decorator(args, data, args.ceritfy_ensemble, random_probs=random_probs)

                # save or load certify_masked_sentence
                index_certify_masked_sentence=-1
                if not os.path.exists(os.path.join(args.save_path,"certify_masked_sentence",f"{index_org_sentence}")):
                    os.makedirs(os.path.join(args.save_path,"certify_masked_sentence",f"{index_org_sentence}"))
                for instance in tmp_instances:
                    index_certify_masked_sentence+=1
                    if instance.text_b is None:
                        certify_masked_sentence_path = os.path.join(args.save_path,"certify_masked_sentence",f"{index_org_sentence}",f"a-{index_certify_masked_sentence}")
                        
                        if args.recover_past_data:
                            if os.path.exists(certify_masked_sentence_path):
                                with open(certify_masked_sentence_path, 'r') as file:
                                    content = file.read()
                                    instance.text_a = content
                        with open(certify_masked_sentence_path, 'w') as file:
                            file.write(instance.text_a)
                    else:
                        raise RuntimeError

                for data in tmp_instances:
                    data.text_a = data.text_a.replace("<mask>", args.mask_word)
                    if data.text_b is not None:
                        data.text_b = data.text_b.replace("<mask>", args.mask_word)

                if args.denoise_method == None:
                    pass
                elif "chatgpt" in args.denoise_method:
                    if not os.path.exists(os.path.join(args.save_path,"certify_denoised_sentence",f"{index_org_sentence}",f"a-0")):
                        denoise_instance(tmp_instances, args)
                elif args.denoise_method == 'alpaca':
                    if not os.path.exists(os.path.join(args.save_path,"certify_denoised_sentence",f"{index_org_sentence}",f"a-0")):
                        alpaca.denoise_instances(tmp_instances)
                elif args.denoise_method == 'roberta':
                    if not os.path.exists(os.path.join(args.save_path,"certify_denoised_sentence",f"{index_org_sentence}",f"a-0")):
                        alpaca.roberta_denoise_instances(tmp_instances)
                elif args.denoise_method == 'remove_mask':
                    if not os.path.exists(os.path.join(args.save_path,"certify_denoised_sentence",f"{index_org_sentence}",f"a-0")):
                        for instance in tmp_instances:
                            instance.text_a = instance.text_a.replace(f"{args.mask_word} ", '').replace(f" {args.mask_word}", '')
                            if instance.text_b is not None:
                                instance.text_b = instance.text_b.replace(f"{args.mask_word} ", '').replace(f" {args.mask_word}", '')

                # save or load certify_denoised_sentence
                if args.denoise_method is not None:
                    if not os.path.exists(os.path.join(args.save_path,"certify_denoised_sentence",f"{index_org_sentence}")):
                        os.makedirs(os.path.join(args.save_path,"certify_denoised_sentence",f"{index_org_sentence}"))
                    index_certify_denoised_sentence=-1
                    for instance in tmp_instances:
                        index_certify_denoised_sentence+=1
                        if instance.text_b is None:
                            certify_denoised_sentence_path = os.path.join(args.save_path,"certify_denoised_sentence",f"{index_org_sentence}",f"a-{index_certify_denoised_sentence}")
                            if args.recover_past_data:
                                if os.path.exists(certify_denoised_sentence_path):
                                    with open(certify_denoised_sentence_path, 'r') as file:
                                        content = file.read()
                                        instance.text_a = content
                            with open(certify_denoised_sentence_path, 'w') as file:
                                file.write(instance.text_a)
                        else:
                            raise RuntimeError

                # load certify_prediction
                if args.recover_past_data:
                    if os.path.exists(os.path.join(args.save_path,"certify_prediction",f"{index_org_sentence}",f"0")):
                        past_certify_predictions = []
                        for i in range(len(tmp_instances)):
                            with open(os.path.join(args.save_path,"certify_prediction",f"{index_org_sentence}",f"{i}"), 'r') as file:
                                content = file.read()
                                past_certify_predictions.append(content)
                    else:
                        past_certify_predictions = None
                else:
                    past_certify_predictions = None

                if args.recover_past_data:
                    if os.path.exists(os.path.join(args.save_path,"certify_prediction_prob",f"{index_org_sentence}.npy")):
                        past_pred_predictions_prob = np.load(os.path.join(args.save_path,"certify_prediction_prob",f"{index_org_sentence}.npy"))
                    else:
                        past_pred_predictions_prob = None
                else:
                    past_pred_predictions_prob = None

                if args.predictor == 'bert':
                    tmp_probs = predictor.predict_batch(tmp_instances)
                    # certify_predictions = None
                else:
                    tmp_probs, certify_predictions = predictor.predict_batch(tmp_instances,past_certify_predictions,past_pred_predictions_prob)
                    # save pred_prediction
                    if not os.path.exists(os.path.join(args.save_path,"certify_prediction",f"{index_org_sentence}")):
                        os.makedirs(os.path.join(args.save_path,"certify_prediction",f"{index_org_sentence}"))
                    
                    if certify_predictions is not None:
                        for i in range(len(certify_predictions)):
                            certify_prediction_path = os.path.join(args.save_path,"certify_prediction",f"{index_org_sentence}",f'{i}')
                            with open(certify_prediction_path, 'w') as file:
                                    file.write(certify_predictions[i])

                    # save certify_prediction_prob
                    np.save(os.path.join(args.save_path,"certify_prediction_prob",f"{index_org_sentence}.npy"), tmp_probs)

                guess_count = np.bincount(np.argmax(tmp_probs, axis=-1), minlength=num_labels)[guess]
                lower_bound, upper_bound = lc_bound(guess_count, args.ceritfy_ensemble, args.alpha)

                guess_counts = np.bincount(np.argmax(tmp_probs, axis=-1), minlength=num_labels)
                print('guess_counts:',guess_counts)
                tmp = guess_counts/guess_count.sum()
                
                entropy_list.append(-tmp*np.log(np.clip(tmp, 1e-6, 1)))
                print("lower_bound:",lower_bound,file=log_file,flush=True)
                if args.certify_lambda:
                    radius = population_radius_for_majority(lower_bound, data_length, keep_nums, lambda_value=guess_count / args.ceritfy_ensemble)
                else:
                    radius = population_radius_for_majority(lower_bound, data_length, keep_nums)
                

    def saving_model_by_epoch(self, args: ClassifierArgs, epoch: int):
        # saving
        if args.saving_step is not None and args.saving_step != 0:
            if (epoch - 1) % args.saving_step == 0:
                self.save_model_to_file(args.saving_dir,
                                        args.build_saving_file_name(description='epoch{}'.format(epoch)))


    def mask_instance_decorator(self, args: ClassifierArgs, instance, numbers:int=1, return_indexes:bool=False,random_probs=None):
        if self.forbidden_words is not None:
            forbidden_index = mask_forbidden_index(instance.perturbable_sentence(), self.forbidden_words)
            return mask_instance(instance, args.sparse_mask_rate, self.tokenizer.mask_token, numbers, return_indexes, forbidden_index,random_probs=random_probs)
        else:
            return mask_instance(instance, args.sparse_mask_rate, self.tokenizer.mask_token, numbers, return_indexes,random_probs=random_probs)


    @classmethod
    def run(cls, args: ClassifierArgs, alpaca=None):
        # build logging
        # including check logging path, and set logging config
        args.build_logging_dir()
        args.build_logging()
        logging.info(args)

        args.build_environment()
        # check dataset and its path
        args.build_dataset_dir()

        args.build_saving_dir()
        args.build_caching_dir()

        if args.dataset_name in ['agnews', 'snli']:
            args.keep_sentiment_word = False

        classifier = cls(args)
        classifier.methods[args.mode](args,alpaca=alpaca)
