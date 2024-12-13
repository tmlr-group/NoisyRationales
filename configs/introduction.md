## Config Introduction

|Category | Parameter | Sub-Parameter | Description |Examples|
|------ | ------ | ------ | ------ | ------ |
|Model|model||llm model name|"gpt-3.5-turbo", "gemini-pro", "mixtral", "llama-2-70b"|
|Dataset|dataset||the dataset used for the experiment.|"base_math", "symbolic", "commonsense"|
||start_num||the starting number of the experiment.| 0 |
||test_num||the number of test instances.|200|
||batch_size||the size of the data processed per batch.|1, 5|
|Task Config|math|subtask|the subtask of Nora-Math|base-9, base-11|
||symbolic|subtask|the subtask of Nora-symbolic|equal, longer|
|Generation|use_processed_dataset||whether use processed dataset, or generate test by detailed setting|True, False|
||processed_dataset_options|processed_dataset_path|processed dataset path or default dataset|processed dataset path or one of ["default-zeroshot"， "default-clean", "default-(irrelevant,inaccurate)-(easy,medium,hard)-(fixed,random)"]|
|||n_shots|shots num|1, 2, 3, 4, 5|
|||using_subset|||
||raw_dataset_options|if_in_context|Represent whether use in-context shot for reasoning.|True, False|
|||n_shots|The number of clean rationale shot|0,1,2,3,4...|
|||if_noise|Represent whether exist noise shots|True, False|
|||n_noisy_shots|The number of noisy rationale shot|1,2,3,4....|
|||noisy_type|The type of noisy rationale shot|irrelevant, inaccurate|
|||noisy_ratio|The ratio of inserting a noise thought after a clean thought.|0-1|
|||noise_distribution|random: each clean thought have the possibility of noisy_ration to get a noisy thought, fixed: each shot have n_clean_thought * ratio of noisy thoughts| random, fixed|
||prefix_context||Represent whether put in-context shots into the prompt prefix or mix as a messages list|True, False|
||method||Represent what kind of method to process the reasoning|CD-CoT, basemodel,  smoothllm, selfdenoise, selfpolish, contrastivecot, ISC, SCO, BT|
||temperature_reason||the reasoning temperature. Available if method is not CD-CoT|0-1|
||n_reason||The reasoning repeat times. Available if method is not CD-CoT|1,2,3,4,5....| -->
<!-- ||CD-CoT|||| -->
<!-- ||gpt|api|version of gpt api|0.28, 1| -->
<!-- |ICL|if_in_context|| symbol of whether use in-context demos |True, False|
||n_shots|| w/o noise shots num | 1, 2, 3|
|Noise|if_noise|symbol of whether use noisy demos|True, False (be False if if_in_context is False)|
||n_noisy_shots| noisy shots num | 1, 2, 3|
||noise_type| type of noise | "irrelavant", "minor-error" |
||noise_ratio| ratio of each thought insert a sentence irrelavant noise or become a minor-error thought|0.2, 0.5, 0.8|
||noise_distribution| fixed noise num in a example shot or just same possibilty to insert noise in each thought| "fixed", "random"| -->
