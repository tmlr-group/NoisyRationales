import yaml
import os
import json
import re
import pickle
import data_process.math.math as nora_math
import data_process.commonsense.commonsense as nora_commonsense
import data_process.symbolic.symbolic as nora_symbolic
import pandas as pd
import nltk
import random
import time
from datetime import datetime
import copy
import string
import argparse
import zipfile
import ast
from dotenv import load_dotenv

parser = argparse.ArgumentParser()
parser.add_argument('-config', type=str, default='config.yml', help='Path to the config file. If the config file is setted, other parameter is unnessary.')
parser.add_argument('-task', type=str, default=None, help='Task type to perform (e.g math_base-9)')
parser.add_argument('-method', type=str, default=None, help='Method name to use (e.g basemodel, CD-CoT, selfconsistency, ISC)')
parser.add_argument('-model', type=str, default=None, help='Model name')
parser.add_argument('-test_num', type=str, default=None, help='Number of test cases to run(e.g 100, 200)')
parser_args = parser.parse_args()
if parser_args.task != None:
    parser_args.config = "quick_start.yml"


def wr_log(obj, log_file):
    print(obj)
    log_file.write(str(obj) + "\n")
    log_file.flush()


class noise_test:
    def __init__(self, args) -> None:
        load_dotenv()
        self.config = args
        self._model_name = args["model"]
        self._dataset_name = args["dataset"]
        self._start_num = args["start_num"]
        self._test_num = args["test_num"]
        self._batch_size = args["batch_size"]
        self.max_token = 0
        assert self._test_num / self._batch_size == int(
            self._test_num / self._batch_size), "test_num / batch_size should be a positive integer"
        
        self.shuffle_study = args["shuffle_study"] if "shuffle_study" in args else False
        if self.shuffle_study:
            self.shuffle_type = args["shuffle_type"]

        self.use_processed_dataset = args["use_processed_dataset"]
        if self.use_processed_dataset or parser_args.task != None:
            processed_dataset_options = args["processed_dataset_options"]
            processed_dataset_path = processed_dataset_options["processed_dataset_path"]
            if parser_args.task != None:
                if parser_args.model != None:
                    self._model_name = parser_args.model
                if parser_args.test_num != None:
                    self._test_num = int(parser_args.test_num)
                labels = parser_args.task.split("_")
                task = labels[0]
                assert task in ["math", "symbolic", "commonsense"]
                self._dataset_name = task
                if task != "commonsense":
                    subtask = labels[1]
                    dataset_label = labels[2:]
                    args[task]["subtask"] = subtask
                else:
                    dataset_label = labels[1:]
                self.processed_dataset_path = self._get_default_processed_dataset_name(dataset_label)
            elif processed_dataset_path.startswith("default-"):
                dataset_label = processed_dataset_path.split("-")[1:]
                self.processed_dataset_path = self._get_default_processed_dataset_name(dataset_label)
            else:
                self.processed_dataset_path = processed_dataset_path
            with open(self.processed_dataset_path, "r", encoding="utf-8") as f:
                config = json.load(f)["config"]
            if dataset_label[0] == "zeroshot":
                config["if_in_context"] = False
            else:
                config["if_in_context"] = True
                if config["if_noise"] == True:
                    if isinstance(processed_dataset_options["n_shots"], int): 
                        config["n_noisy_shots"] = processed_dataset_options["n_shots"]
                        config["n_shots"] = 0
                    else:
                        config["n_noisy_shots"] = int(processed_dataset_options["n_shots"].split("+")[0])
                        config["n_shots"] = int(processed_dataset_options["n_shots"].split("+")[1]) 
                else:
                    config["n_shots"] = processed_dataset_options["n_shots"]
                    config["n_noisy_shots"] = 0
                assert config["n_shots"] + config["n_noisy_shots"]  <= config["n_max_shots"]
        else:
            config = args["raw_dataset_options"]

        self._if_in_context = config["if_in_context"] if "if_in_context" in config else False
        if self._if_in_context:
            self._if_noise = config["if_noise"] if "if_noise" in config else False
            self._n_shots = config["n_shots"] if "n_shots" in config else 1
            self._n_weak_shots = config["n_weak_shots"] if "n_weak_shots" in config else 0
        else:
            self._if_noise = False
            self._n_shots = 0
            self._n_weak_shots = 0

        if self._if_noise:
            self._n_noisy_shots = config["n_noisy_shots"] if "n_noisy_shots" in config else 0
            if self._n_noisy_shots == 0:
                self._if_noise = False
                self._noise_type = None
                self._noise_ratio = 0
                self._noise_semantic_related = 0
                self._noise_distribution = None
            else:
                self._noise_type = config["noise_type"]
                if self._noise_type == "irrelevant":
                    self._noise_semantic_related = config["noise_semantic_related"] if "noise_semantic_related" in config else 0
                else:
                    self._noise_semantic_related = 0
                self._noise_ratio = config["noise_ratio"]
                self._noise_distribution = config["noise_distribution"]
        else:
            self._n_noisy_shots = 0
            self._noise_type = None
            self._noise_ratio = 0
            self._noise_semantic_related = 0
            self._noise_distribution = None

        self._prefix_context = args["prefix_context"] if "prefix_context" in args else False

        random.seed(time.time())

        self._init_model()
        self._init_dataset()
        self._init_method()
        log_name = args["log_name"] if "log_name" in args else self._get_log_file_name()
        print(f"test result is in {log_name}")
        self._log_file_name = log_name
        self._log_file = open(log_name, 'w', encoding='utf-8')

        dirname = os.path.dirname(log_name)
        basename = os.path.basename(log_name)
        name_without_ext = os.path.splitext(basename)[0]
        self._pickle_name = os.path.join(dirname, name_without_ext + '.pkl')

        self._log(args)
        self._correct_num = 0
        self._error_num = 0
        self._not_match_num = 0
        self._case_list = []
        self._noisy_ICL_correct_list = []
        self._reason_ICL_list = []
        self._answers_list = []
        self._contents_list = []
        self._noise_test_result = None
        return

    def _unzip_default_processed_dataset(self, file_dir):
        file_path = os.path.join(file_dir, "processed.zip")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(file_dir)
        print(f"processed_dataset has been extracted to {file_dir}")

    def _get_default_processed_dataset_name(self, dataset_label):
        args = self.config
        noise_type = ["zeroshot", "clean", "irrelevant", "inaccurate"]
        noise_difficulty = ["easy", "medium", "hard"]
        type = dataset_label[0]
        assert type in noise_type
        if type in ["irrelevant", "inaccurate"]:
            file_name = f"{type}"
            difficulty = dataset_label[1]
            if len(dataset_label) > 2:
                distribution = dataset_label[2]
            else:
                distribution = "fixed"
            assert difficulty in noise_difficulty
            file_name += f"_{difficulty}_{distribution}.json"
        else:
            file_name = "clean.json"
        if self._dataset_name == "math":
            subtask = args[self._dataset_name]["subtask"]
            dataset_dir = os.path.join("data", "math")
            processed_dataset_dir = os.path.join("data", "math", "processed", subtask)
        elif self._dataset_name == "commonsense":
            dataset_dir = os.path.join("data", "commonsense")
            processed_dataset_dir = os.path.join("data", "commonsense", "processed")
        elif self._dataset_name == "symbolic":
            subtask = args[self._dataset_name]["subtask"]
            dataset_dir = os.path.join("data", "symbolic")
            processed_dataset_dir = os.path.join("data", "symbolic", "processed", subtask)
        else:
            raise ValueError(f"dataset {self._dataset_name} are not supported in default")
        if not os.path.exists(os.path.join(processed_dataset_dir, file_name)):
            self._unzip_default_processed_dataset(dataset_dir)
        if not os.path.exists(os.path.join(processed_dataset_dir, file_name)):
            raise ValueError(f"default file {os.path.join(processed_dataset_dir, file_name)} not exist")
        return os.path.join(processed_dataset_dir, file_name)

    def _init_model(self):
        if "llama" in self._model_name:
            from llm_model.llama.my_llama import my_llama
            model_config = self.config["llama"] if "llama" in self.config else None
            self._model = my_llama(model=self._model_name, config=model_config)
        elif self._model_name.split("-")[0] == "gpt":
            from llm_model.my_gpt.my_gpt import my_gpt
            model_config = self.config["gpt"] if "gpt" in self.config else None
            self._model = my_gpt(model=self._model_name, config=model_config, prefix_context=self._prefix_context)
        elif self._model_name == "gemini-pro":
            from llm_model.Gemini.my_gemini import my_gemini
            model_config = self.config["gemini"] if "gemini" in self.config else None
            self._model = my_gemini(config=model_config)
        elif self._model_name == "mixtral":
            from llm_model.mixtral.my_mixtral import my_mixtral
            model_config = self.config["my_mixtral"] if "my_mixtral" in self.config else None
            self._model = my_mixtral(config=model_config)
        elif self._model_name.split("-")[0] == "glm":
            from llm_model.zhipu.zhipu import my_zhipu
            model_config = self.config["zhipu"] if "zhipu" in self.config else None
            self._model = my_zhipu(model=self._model_name, config=model_config, prefix_context=self._prefix_context)
        else:
            raise ValueError("Unsupported model {}".format(self._model_name))

    def _load_processed_dataset(self):
        with open(self.processed_dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            dataset_content = dataset["content"]
            if "system-prompt" in dataset:
                self._dataset_system_prompt = dataset["system-prompt"]
            else:
                self._dataset_system_prompt = None
        return dataset_content

    def _init_dataset(self):
        processor_config = self.config[self._dataset_name] if self._dataset_name in self.config else None
        if self._dataset_name == "math":
            self._dataset_processor = nora_math.math(n_shots=self._n_shots,
                                                          n_noisy_shots=self._n_noisy_shots,
                                                          noise_type=self._noise_type, noise_semantic_related = self._noise_semantic_related, noise_ratio=self._noise_ratio,
                                                          noise_distribution=self._noise_distribution,
                                                          prefix_context=self._prefix_context, config=processor_config)
        elif self._dataset_name == "commonsense":
            self._dataset_processor = nora_commonsense.commonsense(n_shots=self._n_shots,
                                                                      n_noisy_shots=self._n_noisy_shots,
                                                                      noise_type=self._noise_type,
                                                                      noise_ratio=self._noise_ratio,
                                                                      prefix_context=self._prefix_context,
                                                                      config=processor_config)
            self._dataset_config = self._dataset_processor.get_config()
        elif self._dataset_name == "symbolic":
            self._dataset_processor = nora_symbolic.symbolic(n_shots=self._n_shots, n_noisy_shots=self._n_noisy_shots,
                                                              noise_type=self._noise_type,
                                                              noise_semantic_related = self._noise_semantic_related,
                                                              noise_ratio=self._noise_ratio,
                                                              noise_distribution=self._noise_distribution,
                                                              prefix_context=self._prefix_context,
                                                              config=processor_config)
        else:
            raise ValueError("Unsupported dataset {}".format(self._dataset_name))
        if not self.use_processed_dataset:
            self._dataset = self._dataset_processor.load_data()
        else:
            self._dataset_processor.load_data()
            self._dataset = self._load_processed_dataset()
        assert len(self._dataset) >= self._test_num

    def _get_logged_rephrased_result_file(self):
        dir_name = os.path.join(self.log_dir, "rephrased")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if self._if_noise:
            if self._noise_type == "irrelevant":
                file_name = f"{self._noise_type}_{self._noise_ratio}_sem{self._noise_semantic_related}_{self._noise_distribution}_{self._start_num}_{self._test_num}_n{self.CDCoT.n_rephrase}_t{self.CDCoT.temperature_rephrase}_p{self.CDCoT.topp_rephrase}.json"
            else:
                file_name = f"{self._noise_type}_{self._noise_ratio}_{self._noise_distribution}_{self._start_num}_{self._test_num}_n{self.CDCoT.n_rephrase}_t{self.CDCoT.temperature_rephrase}_p{self.CDCoT.topp_rephrase}.json"
        else:
            file_name = f"clean_{self._start_num}_{self._test_num}_n{self.CDCoT.n_rephrase}_t{self.CDCoT.temperature_rephrase}_p{self.CDCoT.topp_rephrase}.json"
        return os.path.join(dir_name, file_name)

    def _get_logged_ICL_list_file(self):
        dir_name = os.path.join(self.log_dir, "icl")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if self._if_noise:
            if self._noise_type == "irrelevant":
                file_name = f"{self._noise_type}_{self._noise_ratio}_sem{self._noise_semantic_related}_{self._noise_distribution}_{self._start_num}_{self._test_num}_n{self.CDCoT.n_rephrase}_t{self.CDCoT.temperature_rephrase}_p{self.CDCoT.topp_rephrase}_m{self.CDCoT.m_select}_ICL.json"
            else:
                file_name = f"{self._noise_type}_{self._noise_ratio}_{self._noise_distribution}_{self._start_num}_{self._test_num}_n{self.CDCoT.n_rephrase}_t{self.CDCoT.temperature_rephrase}_p{self.CDCoT.topp_rephrase}_m{self.CDCoT.m_select}_ICL.json"
        else:
            file_name = f"clean_{self._start_num}_{self._test_num}_n{self.CDCoT.n_rephrase}_t{self.CDCoT.temperature_rephrase}_p{self.CDCoT.topp_rephrase}_m{self.CDCoT.m_select}_ICL.json"
        return os.path.join(dir_name, file_name)

    def _init_method(self):
        if parser_args.method == None:
            self.method = self.config["method"]
        else:
            self.method = parser_args.method
        args = self.config
        if self.method == "basemodel" or self.method == "selfconsistency":
            self.temperature_reason = args["temperature_reason"] if "temperature_reason" in args else 1
            self.n_reason = args["n_reason"] if "n_reason" in args else 1
        elif self.method == "CD-CoT":
            from method.CD_CoT import CDCoT
            self.use_logged_rephrased_result = args[
                "use_logged_rephrased_result"] if "use_logged_rephrased_result" in args else False
            self.use_logged_ICL_result = args["use_logged_ICL_result"] if "use_logged_ICL_result" in args else False
            n_rephrase = args["n_rephrase"] if "n_rephrase" in args else 5
            temperature_rephrase = args["temperature_rephrase"] if "temperature_rephrase" in args else 1
            topp_rephrase = args["topp_rephrase"] if "topp_rephrase" in args else 1
            use_clean_shot = args["use_clean_shot"] if "use_clean_shot" in args else True
            c_reason = args["c_reason"] if "c_reason" in args else [5]
            m_select = len(c_reason)
            temp_reason = args["temp_reason"] if "temp_reason" in args else 1
            topp_reason = args["topp_reason"] if "topp_reason" in args else 1
            self.CDCoT = CDCoT.CDCoT(n_rephrase, temperature_rephrase, topp_rephrase, m_select,
                               c_reason, temp_reason, topp_reason, self._model, use_clean_shot)
        elif self.method == "smoothllm":
            from method.smooth_llm_main.lib.defenses import SmoothLLM
            self.temperature_reason = args["temperature_reason"] if "temperature_reason" in args else 1
            self.n_reason = args["n_reason"] if "n_reason" in args else 1
            self.smoothllm = SmoothLLM(self._model, self._dataset_processor, "RandomSwapPerturbation", 10,
                                       self.n_reason)
        elif self.method == "selfdenoise":
            from method.SelfDenoise_main.baseline_test import SelfDenoise
            self.temperature_reason = args["temperature_reason"] if "temperature_reason" in args else 1
            self.n_reason = args["n_reason"] if "n_reason" in args else 1
            self.SelfDenoise = SelfDenoise(n_reason=self.n_reason)
        elif self.method == "contrastivecot":
            self.temperature_reason = args["temperature_reason"] if "temperature_reason" in args else 1
            self.n_reason = args["n_reason"] if "n_reason" in args else 1
        elif self.method == "ISC":
            self.temperature_reason = args["temperature_reason"] if "temperature_reason" in args else 1
            self.n_reason = args["n_reason"] if "n_reason" in args else 1
        elif self.method == "SCO":
            self.temperature_reason = args["temperature_reason"] if "temperature_reason" in args else 1
            self.n_reason = args["n_reason"] if "n_reason" in args else 1
        elif self.method == "selfpolish":
            self.temperature_reason = args["temperature_reason"] if "temperature_reason" in args else 1
            self.n_reason = args["n_reason"] if "n_reason" in args else 1
        elif self.method == "BT":
            self.temperature_reason = args["temperature_reason"] if "temperature_reason" in args else 1
            self.n_reason = args["n_reason"] if "n_reason" in args else 1

    def _get_log_file_name(self):
        log_path = os.path.join("result", self._dataset_name)
        dataset_config = self.config[self._dataset_name] if self._dataset_name in self.config else None
        if dataset_config != None:
            if "subtask" in dataset_config:
                log_path = os.path.join(log_path, dataset_config["subtask"])

        if self._dataset_name == "commonsense":
            if self._dataset_config["subtask"] == "symbolic":
                log_path = os.path.join(log_path, "hop" + str(self._dataset_config["hop"]))
        log_path = os.path.join(log_path, self._model_name)
        log_path = os.path.join(log_path, f"method_{self.method}")
        if "subfolder_suffix_path" in self.config:
            if len(self.config["subfolder_suffix_path"]) > 0:
                log_path = os.path.join(log_path, self.config["subfolder_suffix_path"])
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = "log"
        if self._if_in_context:
            if self._prefix_context:
                log_file += "_prefix"
            log_file += "_ICL_{}clean".format(self._n_shots)
            if self._n_weak_shots > 0:
                log_file += "_{}weak".format(self._n_weak_shots)
            if self.shuffle_study == True:
                log_file += "_shuffle{}".format(self.shuffle_type)
        
        if self._if_noise:
            if self._noise_type == "irrelevant":
                log_file += "_noise_{}{}_sem{}_{}_ratio{}".format(self._n_noisy_shots, self._noise_type, self._noise_semantic_related, self._noise_distribution,
                                                        self._noise_ratio)
            else:
                log_file += "_noise_{}{}_{}_ratio{}".format(self._n_noisy_shots, self._noise_type, self._noise_distribution,
                                                        self._noise_ratio)
        else:
            log_file += "_origin"

        log_file += "_case{}".format(self._test_num)
        if self.method == "basemodel" or self.method == "selfconsistency":
            log_file += "_temp{}_n{}".format(self.temperature_reason, self.n_reason)
        elif self.method == "CD-CoT":
            log_file += "_use_{}".format(self.use_logged_rephrased_result)
            log_file += "_n{}_t{}_p{}".format(self.CDCoT.n_rephrase, self.CDCoT.temperature_rephrase,
                                              self.CDCoT.topp_rephrase)
            log_file += "_m{}_clean_{}".format(self.CDCoT.m_select, self.CDCoT.use_clean_shot)
            log_file += "_c{}_t{}_p_{}".format(len(self.CDCoT.c_reason), self.CDCoT.temp_reason, self.CDCoT.topp_reason)
        else:
            log_file += "_temp{}_n{}".format(self.temperature_reason, self.n_reason)
        log_file += ".log"
        log_file_path = os.path.join(log_path, log_file)
        self.log_dir = log_path
        return log_file_path

    def run(self):
        self._log("Start time: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        if self._noise_test_result is None:
            test_num = self._test_num
            if isinstance(self._dataset, pd.DataFrame):
                data_iter = self._dataset.iterrows()
            else:
                data_iter = enumerate(self._dataset)
            for count, raw_data in data_iter:
                if count < self._start_num:
                    continue
                self._question_insert(raw_data)
                test_num -= 1
                if test_num <= 0:
                    break
            
            if self.method == "CD-CoT":
                if self.use_processed_dataset and self.use_logged_ICL_result:
                    self._load_ICL_list_result()
                    self._CD_CoT_with_ICL_list()  # ablation clean_shot
                elif self.use_processed_dataset and self.use_logged_rephrased_result:
                    self._load_rephrased_result()
                    self._CD_CoT()  # ablation N M C
                else:
                    self._record_rephrase_result()
                    self._CD_CoT()
            else:
                self._query_process()
            self._noise_test_result = dict()
            self._noise_test_result["correct_num"] = self._correct_num
            self._noise_test_result["error_num"] = self._error_num
            self._noise_test_result["noisy_ICL_correct_list"] = self._noisy_ICL_correct_list
            self._noise_test_result["not_match_num"] = self._not_match_num
            self._noise_test_result["answers_list"] = self._answers_list
            self._noise_test_result["contents_list"] = self._contents_list
            self._noise_test_result["question_list"] = [case["question"] for case in self._case_list]
            self._save_result()
            self._log("correct_num:{}, error_num:{}, Acc:{}".format(self._correct_num, self._error_num,
                                                                    self._correct_num / (
                                                                            self._correct_num + self._error_num + self._not_match_num)))
        self._log("End time: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        return self._noise_test_result

    def _log(self, obj):
        print(obj)
        self._log_file.write(str(obj) + "\n")
        self._log_file.flush()

    def _response_process(self, case_batch):
        for case in case_batch:
            context = case["messages"]
            label = case["label"] if "label" in case else None
            self._log(json.dumps(context))
            if label:
                self._log("\nCorrect answer is {}\n".format(label))
            responses = case["messages"][-1]  # all responses
            for response in responses:
                raw_answer = response["content"]
                self._contents_list.append(raw_answer)
                self._log(raw_answer)

                answer = self._dataset_processor.match_answer(raw_answer)
                self._log("match answer: {}".format(answer))
                if answer:
                    if answer == label:
                        self._log("right")
                        self._correct_num += 1
                        self._answers_list.append([answer, 1])
                    else:
                        self._log("wrong")
                        self._error_num += 1
                        self._answers_list.append([answer, 0])
                else:

                    self._log("not match")
                    self._not_match_num += 1
                    self._answers_list.append("not match")
        return

    def _query_process(self):
        batch_size = self._batch_size
        case_list = [copy.deepcopy(self._case_list[i:i + batch_size]) for i in
                     range(0, len(self._case_list), batch_size)]
        for index, case_batch in enumerate(case_list):
            if self.method == "basemodel" or self.method == "selfconsistency":
                case_n = self.n_reason
                self._model.query_case_batch(case_batch, self.temperature_reason, self.n_reason)
                self._response_process(case_batch)
            elif self.method == "smoothllm":
                case_batch = self.smoothllm(case_batch)
                self._response_process(case_batch)
                case_n = 1
            elif self.method == "selfdenoise":
                case_batch = self.SelfDenoise.certify(case_batch, model=self._model, log_file=self._log_file)
                self._response_process(case_batch)
                case_n = self.n_reason
            elif self.method == "contrastivecot":
                if self._dataset_name == "math":
                    expr = "47+58"
                elif self._dataset_name == "symbolic":
                    expr = ["walk around right twice after run opposite left",
                            ["I_TURN_LEFT", "I_TURN_LEFT", "I_RUN", "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT", "I_WALK",
                             "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT", "I_WALK",
                             "I_TURN_RIGHT",
                             "I_WALK", "I_TURN_RIGHT", "I_WALK", "I_TURN_RIGHT", "I_WALK"]]
                elif self._dataset_name == "commonsense":
                    expr = self._dataset_processor.get_demos(1)[0]
                postive_QAL = []
                postive_QAL.append(self._dataset_processor.get_question(expr))
                postive_QAL.append(self._dataset_processor.get_correct_answer(expr))
                postive_QAL.append(self._dataset_processor.get_label(expr))
                from method.Contrastive_CoT.Contrastive_CoT import Contrastive_CoT
                case_batch = Contrastive_CoT(postive_QAL=postive_QAL, case_batch=case_batch, model=self._model,
                                             dataprocessor=self._dataset_processor, n_reason=self.n_reason)
                self._response_process(case_batch)
                case_n = self.n_reason
            elif self.method == "ISC":
                from method.Intrinsic_Self_Correct.Intrinsic_Self_Correct import Intrinsic_Self_Correct
                case_batch = Intrinsic_Self_Correct(case_batch=case_batch, model=self._model,
                                                    dataset_name=self._dataset_name, n_reason=self.n_reason)
                self._response_process(case_batch)
                case_n = self.n_reason
            elif self.method == "SCO":
                from method.Intrinsic_Self_Correct.Intrinsic_Self_Correct import Intrinsic_Self_Correct
                case_batch = Intrinsic_Self_Correct(case_batch=case_batch, model=self._model,
                                                    dataset_name=self._dataset_name, n_reason=self.n_reason,
                                                    answer_match_func=self._dataset_processor.match_answer,
                                                    method="SCO")
                self._response_process(case_batch)
                case_n = self.n_reason
            elif self.method == "selfpolish":
                from method.SelfPolish.selfpolish import SelfPolish
                SP = SelfPolish(model=self._model, temp=self.temperature_reason)
                case_batch = SP.polish_batch(case_batch)
                self._model.query_case_batch(case_batch, self.temperature_reason, self.n_reason)
                self._response_process(case_batch)
                case_n = self.n_reason
            elif self.method == "BT":
                from method.backtrace.backtrace import backtrace
                case_batch = backtrace(case_batch=case_batch, model=self._model)
                self._model.query_case_batch(case_batch, temperature=self.temperature_reason, n=self.n_reason)
                self._response_process(case_batch)
                case_n = self.n_reason
            elif self.method == "SCO":
                from method.Intrinsic_Self_Correct.Intrinsic_Self_Correct import Intrinsic_Self_Correct
                case_batch = Intrinsic_Self_Correct(case_batch=case_batch, model=self._model,
                                                    dataset_name=self._dataset_name, n_reason=self.n_reason,
                                                    answer_match_func=self._dataset_processor.match_answer,
                                                    method="SCO")
                self._response_process(case_batch)
                case_n = self.n_reason
            elif self.method == "selfpolish":
                from method.SelfPolish.selfpolish import SelfPolish
                SP = SelfPolish(model=self._model, temp=self.temperature_reason)
                case_batch = SP.polish_batch(case_batch)
                self._model.query_case_batch(case_batch, self.temperature_reason, self.n_reason)
                self._response_process(case_batch)
                case_n = self.n_reason
            elif self.method == "BT":
                from method.backtrace.backtrace import backtrace
                case_batch = backtrace(case_batch=case_batch, model=self._model)
                self._model.query_case_batch(case_batch, temperature=self.temperature_reason, n=self.n_reason)
                self._response_process(case_batch)
                case_n = self.n_reason
            self._log(
                f"index {index}/{len(case_list) - 1}, correct_num {self._correct_num}, error_num {self._error_num}, not match {self._not_match_num}, "
                f"Acc {self._correct_num / (self._correct_num + self._error_num+self._not_match_num)}")
            if not self._model_name.startswith("gemini"):
                self._log(self._model.compute_cost())

        self._answers_list = [self._answers_list[i:i + case_n]
                              for i in range(0, len(self._answers_list), case_n)]
        self._contents_list = [self._contents_list[i:i + case_n]
                               for i in range(0, len(self._contents_list), case_n)]

    def _question_insert(self, raw_data):
        if not self.use_processed_dataset:
            processed_case = self._dataset_processor.get_case(raw_data)
            self._case_list.append(processed_case)
        else:
            case = dict()
            case["question"] = raw_data["question"]
            case["label"] = raw_data["label"]
            demos = []
            if self.method == "BT":
                case["first_error_position_list"] = []
            
            if self._if_noise == True:
                for i in range(self._n_noisy_shots):
                    demo = [raw_data["CoT_demos"][i]["question"], raw_data["CoT_demos"][i]["answer"]]
                    demos.append(demo)
                    if self.method == "BT":
                        sentence_with_noise_list = ast.literal_eval(raw_data["CoT_demos"][i]["sentences_with_noise"])
                        if 1 in sentence_with_noise_list:
                            case["first_error_position_list"].append(sentence_with_noise_list.index(1))
                        else:
                            case["first_error_position_list"].append(-1)
                for i in range(self._n_shots):
                    if self._dataset_name == "commonsense":
                        expr = self._dataset_processor.get_demos(1)[0]
                    elif self._dataset_name == "symbolic":
                        expr = self._dataset_processor.get_demos(1)[0]
                    elif self._dataset_name == "math":
                        index = i + self._n_noisy_shots
                        question = raw_data["CoT_demos"][index]["question"]
                        pattern = r'[\da-fA-F]+\+[\da-fA-F]+'
                        match = re.search(pattern, question)
                        expr =  match.group()
                    demo = [self._dataset_processor.get_question(expr), self._dataset_processor.get_correct_answer(expr)]
                    demos.append(demo)
                    random.shuffle(demos)
                    if self.method == "BT":
                        case["first_error_position_list"].append(-1)
            else:
                for i in range(self._n_shots + self._n_noisy_shots):
                    demo = [raw_data["CoT_demos"][i]["question"], raw_data["CoT_demos"][i]["answer"]]
                    demos.append(demo)
                    if self.method == "BT":
                        sentence_with_noise_list = ast.literal_eval(raw_data["CoT_demos"][i]["sentences_with_noise"])
                        if 1 in sentence_with_noise_list:
                            case["first_error_position_list"].append(sentence_with_noise_list.index(1))
                        else:
                            case["first_error_position_list"].append(-1)
            case["in-context"] = demos
            if self._dataset_system_prompt is not None:
                case["system-prompt"] = self._dataset_system_prompt
            self._case_list.append(case)
        
        if self.shuffle_study == True:
            rationale_label_list = []
            for case in self._case_list:
                if "in-context" in case:
                    for shot in case["in-context"]:
                        answer = shot[1]
                        reversed_sentence = answer[::-1]
                        first_period_idx = reversed_sentence.find(".")
                        second_period_idx = reversed_sentence.find(".", first_period_idx + 1)
                        last_sentence_idx = len(reversed_sentence) - second_period_idx + 1
                        rationale = answer[:last_sentence_idx]
                        label = answer[last_sentence_idx:]
                        rationale_label_list.append([rationale, label])
                    
                    last_rationale = copy.deepcopy(rationale_label_list[-1][0])
                    last_label = copy.deepcopy(rationale_label_list[-1][1])
                    for i in reversed(range(len(rationale_label_list))):
                        if i == 0:
                            if self.shuffle_type == 1:
                                rationale_label_list[i][0] = last_rationale
                            elif self.shuffle_type == 2:
                                rationale_label_list[i][1] = last_label
                            elif self.shuffle_type == 3:
                                rationale_label_list[i][0] = last_rationale
                                rationale_label_list[i][1] = last_label
                        else:
                            if self.shuffle_type == 1:
                                rationale_label_list[i][0] = rationale_label_list[i-1][0]
                            elif self.shuffle_type == 2:
                                rationale_label_list[i][1] = rationale_label_list[i-1][1]
                            elif self.shuffle_type == 3:
                                rationale_label_list[i][0] = rationale_label_list[i-1][0]
                                rationale_label_list[i][1] = rationale_label_list[i-1][1]
                    
                    for i, shot in enumerate(case["in-context"]):
                        shot[1] = rationale_label_list[i][0] + rationale_label_list[i][1]                    
        
        return

    def _save_result(self):
        with open(self._pickle_name, 'wb') as f:
            pickle.dump(self._noise_test_result, f)

    def _load_rephrased_result(self):
        logged_rephrased_result_file = self._get_logged_rephrased_result_file()
        if os.path.exists(logged_rephrased_result_file):
            with open(logged_rephrased_result_file, 'r') as rephrased_result_f:
                noisy_ICL_correct_recording = json.load(rephrased_result_f)
                self.CDCoT.clean_shot = noisy_ICL_correct_recording["clean_shot"]
                self._noisy_ICL_correct_list = noisy_ICL_correct_recording["noisy_ICL_correct_process"]
                rephrased_result_f.close()
        else:
            raise ValueError("Such logged_rephrased_result_file doesn't exist! ")

    def _record_rephrase_result(self):
        json_name = self._get_logged_rephrased_result_file()
        noisy_ICL_correct_recording = dict()
        for case in self._case_list:
            noisy_ICL_correct_object, clean_shot = self.CDCoT.rephrase_icl_shots(case, self._dataset_name,
                                                                                 self._dataset_processor)
            self._log("noisy_ICL_correct_process:\n")
            self._log(noisy_ICL_correct_object)
            if not self._model_name.startswith("gemini"):
                self._log(self._model.compute_cost())
            self._noisy_ICL_correct_list.append(noisy_ICL_correct_object)
        noisy_ICL_correct_recording["clean_shot"] = clean_shot
        noisy_ICL_correct_recording["noisy_ICL_correct_process"] = self._noisy_ICL_correct_list
        with open(json_name, 'w') as rephrase_record_file:
            json.dump(noisy_ICL_correct_recording, rephrase_record_file)
            rephrase_record_file.close()

    def _CD_CoT(self):
        for i in range(self._test_num - self._start_num):
            case = self._case_list[i]
            ICL_correct_process = self._noisy_ICL_correct_list[i]
            select_shot_list, self._reason_ICL_list, n_case = self.CDCoT.CD_CoT(case, ICL_correct_process,
                                                                                self._reason_ICL_list,
                                                                                self._dataset_processor)
            self._log("selected shots list:")
            self._log(select_shot_list)
            self._response_process(n_case)
            self._log(
                f"index {i}/{self._test_num - 1}, correct_num {self._correct_num}, error_num {self._error_num}, "
                f"Acc {self._correct_num / (self._correct_num + self._error_num + self._not_match_num)}")
            if not self._model_name.startswith("gemini"):
                self._log(self._model.compute_cost())
        with open(self._get_logged_ICL_list_file(), 'w', encoding='utf-8') as ICL_file:
            json.dump({"reason_ICL_list": self._reason_ICL_list}, ICL_file)
            ICL_file.close()
        self._answers_list = [self._answers_list[i:i + sum(self.CDCoT.c_reason)]
                              for i in range(0, len(self._answers_list), sum(self.CDCoT.c_reason))]
        self._contents_list = [self._contents_list[i:i + sum(self.CDCoT.c_reason)]
                               for i in range(0, len(self._contents_list), sum(self.CDCoT.c_reason))]

    def _load_ICL_list_result(self):
        logged_ICL_list_result_file = self._get_logged_ICL_list_file()
        if os.path.exists(logged_ICL_list_result_file):
            with open(logged_ICL_list_result_file, 'r') as ICL_result_f:
                ICL_list_recording = json.load(ICL_result_f)
                self._reason_ICL_list = ICL_list_recording["reason_ICL_list"]
                ICL_result_f.close()
        else:
            raise ValueError("Such reason_ICL_list_file doesn't exist! ")

    def _CD_CoT_with_ICL_list(self):
        for i in range(self._test_num - self._start_num):
            case = self._case_list[i]
            ICL_list = self._reason_ICL_list[i]
            n_case = self.CDCoT.CD_CoT_with_ICL_list(case, ICL_list)
            self._response_process(n_case)
            self._log(
                f"index {i}/{self._test_num - 1}, correct_num {self._correct_num}, error_num {self._error_num}, "
                f"Acc {self._correct_num / (self._correct_num + self._error_num + self._not_match_num)}")
            if not self._model_name.startswith("gemini"):
                self._log(self._model.compute_cost())
        self._answers_list = [self._answers_list[i:i + sum(self.CDCoT.c_reason)]
                              for i in range(0, len(self._answers_list), sum(self.CDCoT.c_reason))]
        self._contents_list = [self._contents_list[i:i + sum(self.CDCoT.c_reason)]
                               for i in range(0, len(self._contents_list), sum(self.CDCoT.c_reason))]

    def COT_SC_correct_rate(self, answers_list):

        from collections import Counter
        valid_count = 0
        all_count = 0
        SC_right_count = 0
        for answers in answers_list:
            answers = [sublist for sublist in answers if isinstance(sublist, list)]  # clean answers
            all_count += 1
            if len(answers) == 0:
                continue
            else:
                valid_count += 1

            second_elements_are_1 = [sublist[1] == 1 for sublist in answers]
            any_second_element_is_1 = any(second_elements_are_1)
            if not any_second_element_is_1:
                continue
            true_answer = next((sublist[0] for sublist in answers if sublist[1] == 1), None)
            counter = Counter(sublist[0] for sublist in answers)
            guess_value, _ = counter.most_common(1)[0]
            if guess_value == true_answer:
                SC_right_count += 1
        self._log("SC_correct_num:{}, valid_num:{}, SC_correct_rate:{}".format(SC_right_count, all_count,
                                                                               SC_right_count / all_count))
        return SC_right_count, valid_count, all_count


if __name__ == "__main__":
    config_path = parser_args.config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    test = noise_test(args=config)

    noise_test_result = test.run()
    if test.method in ["selfconsistency", "CD-CoT", "smoothllm", "selfdenoise"]:
        test.COT_SC_correct_rate(noise_test_result["answers_list"])
