# run_search_o1.py
import os
import json
import time
import re
from tqdm import tqdm
import numpy as np
import torch
import string
from typing import Optional, Tuple, List, Dict
import argparse

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from bing_search import (
    web_search, 
    extract_relevant_info, 
    fetch_page_content, 
    extract_snippet_with_context
)
from evaluate import (
    run_evaluation, 
    extract_answer
)
from prompts import (
    get_gpqa_search_o1_instruction, 
    get_math_search_o1_instruction, 
    get_code_search_o1_instruction, 
    get_singleqa_search_o1_instruction, 
    get_multiqa_search_o1_instruction, 
    get_webpage_to_reasonchain_instruction,
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
)
from controller_integration import (
    apply_controller_result,
    bootstrap_controller_import,
    create_controller,
)
from baselines import RandomController, StopOnlyController, TokenBudgetController

# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Search O1 for various datasets and models.")

    # Dataset and split configuration
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        choices=['gpqa', 'math500', 'aime', 'amc', 'livecode', 'nq', 'triviaqa', 'hotpotqa', '2wiki', 'musique', 'bamboogle'],
        help="Name of the dataset to use."
    )

    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['test', 'diamond', 'main', 'extended'],
        help="Dataset split to use."
    )
    parser.add_argument(
        '--custom_data_path',
        type=str,
        default=None,
        help="Optional custom JSON path (list of {task_id, question, ground_truth}). "
             "When provided, skip default dataset file loading and use this file directly.",
    )


    parser.add_argument(
        '--subset_num',
        type=int,
        default=-1,
        help="Number of examples to process. Defaults to all if not specified."
    )

    # Search and document retrieval configuration
    parser.add_argument(
        '--max_search_limit',
        type=int,
        default=10,
        help="Maximum number of searches per question."
    )

    parser.add_argument(
        '--max_turn',
        type=int,
        default=15,
        help="Maximum number of turns."
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help="Maximum number of search documents to return."
    )

    parser.add_argument(
        '--max_doc_len',
        type=int,
        default=3000,
        help="Maximum length of each searched document."
    )

    parser.add_argument(
        '--use_jina',
        type=bool,
        default=True,
        help="Whether to use Jina API for document fetching."
    )

    parser.add_argument(
        '--jina_api_key',
        type=str,
        default='None',
        help="Your Jina API Key to Fetch URL Content."
    )

    # Model configuration
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the pre-trained model."
    )

    # Sampling parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help="Sampling temperature."
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.8,
        help="Top-p sampling parameter."
    )

    parser.add_argument(
        '--top_k_sampling',
        type=int,
        default=20,
        help="Top-k sampling parameter."
    )

    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=None,
        help="Repetition penalty. If not set, defaults based on the model."
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=32768,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    # Search backend configuration
    parser.add_argument(
        '--search_backend',
        type=str,
        default='duckduckgo',
        choices=['bing', 'duckduckgo'],
        help="Search backend: 'duckduckgo' (no API key) or 'bing'."
    )

    parser.add_argument(
        '--bing_subscription_key',
        type=str,
        default=None,
        help="Bing Search API subscription key (required when search_backend=bing)."
    )

    parser.add_argument(
        '--bing_endpoint',
        type=str,
        default="https://api.bing.microsoft.com/v7.0/search",
        help="Bing Search API endpoint."
    )

    # [Controller Integration] Deliberation Controller flags
    parser.add_argument(
        '--controller_enabled',
        type=str2bool,
        default=False,
        help="Enable online Deliberation Controller interventions.",
    )
    parser.add_argument(
        '--controller_project_path',
        type=str,
        default="/data/wanghy/agent_traj",
        help="Project root containing deliberation_controller package.",
    )
    parser.add_argument(
        '--controller_path',
        type=str,
        default="/data/wanghy/agent_traj/deliberation_controller/checkpoints/best_controller.pt",
        help="Path to controller checkpoint.",
    )
    parser.add_argument(
        '--ref_dist_path',
        type=str,
        default="/data/wanghy/agent_traj/deliberation_controller/data/ref_dist_hotpotqa_qwen3.json",
        help="Path to reference distribution JSON for normalization.",
    )
    parser.add_argument(
        '--gate_threshold',
        type=float,
        default=0.5,
        help="Gate threshold for intervention decision.",
    )
    parser.add_argument(
        '--baseline_type',
        type=str,
        default='none',
        choices=['none', 'random', 'token_budget', 'stop_only'],
        help="Use a baseline controller instead of trained controller for comparison.",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Extract arguments
    dataset_name = args.dataset_name
    split = args.split
    custom_data_path = args.custom_data_path
    subset_num = args.subset_num
    MAX_SEARCH_LIMIT = args.max_search_limit
    MAX_TURN = args.max_turn
    top_k = args.top_k
    max_doc_len = args.max_doc_len
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k_sampling = args.top_k_sampling
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens
    search_backend = args.search_backend
    bing_subscription_key = args.bing_subscription_key
    bing_endpoint = args.bing_endpoint
    use_jina = args.use_jina
    jina_api_key = args.jina_api_key
    controller_enabled = bool(args.controller_enabled)
    controller_project_path = args.controller_project_path
    controller_path = args.controller_path
    ref_dist_path = args.ref_dist_path
    gate_threshold = float(args.gate_threshold)
    baseline_type = str(args.baseline_type).lower()

    if search_backend == 'bing' and not bing_subscription_key:
        raise ValueError("--bing_subscription_key is required when --search_backend=bing")

    # [Controller Integration] Optional bootstrap for external controller package.
    controller_factory = None
    runtime_controller_enabled = controller_enabled or (baseline_type != "none")
    if baseline_type != "none":
        if baseline_type == "random":
            controller_factory = lambda: RandomController(intervention_rate=0.1)
        elif baseline_type == "token_budget":
            controller_factory = lambda: TokenBudgetController(budget=6000)
        elif baseline_type == "stop_only":
            bootstrap_controller_import(controller_project_path)
            controller_factory = lambda: StopOnlyController(
                controller_path=controller_path,
                ref_dist_path=ref_dist_path,
                gate_threshold=gate_threshold,
                deliberation_project_path=controller_project_path,
                signal_dim=5,
                num_steps=5,
            )
    elif controller_enabled:
        bootstrap_controller_import(controller_project_path)
        controller_factory = lambda: create_controller(
            controller_path=controller_path,
            ref_dist_path=ref_dist_path,
            gate_threshold=gate_threshold,
            k=5,
            signal_dim=5,
        )
    
    # Adjust parameters based on dataset
    if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'medmcqa', 'pubhealth']:
        MAX_SEARCH_LIMIT = 5
        if dataset_name in ['hotpotqa', 'musique', 'bamboogle', '2wiki']:
            MAX_SEARCH_LIMIT = 10
            MAX_TURN = 15
        top_k = 10
        max_doc_len = 3000
    
    if args.jina_api_key == 'None':
        jina_api_key = None

    # Set default repetition_penalty if not provided
    if repetition_penalty is None:
        repetition_penalty = 1.05 if 'qwq' in model_path.lower() else 1.0

    # Data paths based on dataset
    if custom_data_path:
        data_path = custom_data_path
    elif dataset_name == 'livecode':
        data_path = f'./data/LiveCodeBench/{split}.json'
    elif dataset_name in ['math500', 'gpqa', 'aime', 'amc']:
        data_path = f'./data/{dataset_name.upper()}/{split}.json'
    else:
        data_path = f'./data/QA_Datasets/{dataset_name}.json'

    print('-----------------------')
    if custom_data_path:
        print(f'Using custom dataset file: {custom_data_path}')
        print(f'Base mode: {dataset_name} {split} (prompt/eval/output logic unchanged)')
    else:
        print(f'Using {dataset_name} {split} set.')
    print('-----------------------')

    # ---------------------- Caching Mechanism ----------------------
    # Define cache directories and file paths
    cache_dir = './cache'
    search_cache_path = os.path.join(cache_dir, 'search_cache.json')
    url_cache_path = os.path.join(cache_dir, 'url_cache.json')

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Load existing caches or initialize empty dictionaries
    if os.path.exists(search_cache_path):
        with open(search_cache_path, 'r', encoding='utf-8') as f:
            search_cache = json.load(f)
    else:
        search_cache = {}

    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache = json.load(f)
    else:
        url_cache = {}

    # Function to save caches
    def save_caches():
        with open(search_cache_path, 'w', encoding='utf-8') as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)
        with open(url_cache_path, 'w', encoding='utf-8') as f:
            json.dump(url_cache, f, ensure_ascii=False, indent=2)

    # ---------------------- Model Loading ----------------------
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Define output directory based on model and dataset
    if 'qwq' in model_path.lower():
        if dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode']:
            output_dir = f'./outputs/{dataset_name}.qwq.search_o1'
            if dataset_name == 'gpqa' and (MAX_SEARCH_LIMIT != 5 or top_k != 10):
                output_dir = f'./outputs/runs.analysis/{dataset_name}.qwq.search_o1.{MAX_SEARCH_LIMIT}.{top_k}'
        else:
            output_dir = f'./outputs/runs.qa/{dataset_name}.qwq.search_o1'
    else:
        model_short_name = model_path.split('/')[-1].lower().replace('-instruct', '')
        output_dir = f'./outputs/runs.baselines/{dataset_name}.{model_short_name}.search_o1'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the LLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
    )

    # ---------------------- Data Loading ----------------------
    with open(data_path, 'r', encoding='utf-8') as json_file:
        filtered_data = json.load(json_file)

    # Normalize custom input rows so downstream logic can reuse existing code paths.
    if custom_data_path:
        if not isinstance(filtered_data, list):
            raise ValueError(
                "--custom_data_path expects a JSON list of dicts with task_id/question/ground_truth."
            )
        normalized_rows = []
        for idx, item in enumerate(filtered_data):
            if not isinstance(item, dict):
                raise ValueError(f"Custom data row {idx} is not a dict: {type(item)}")

            question = str(item.get("question", item.get("Question", ""))).strip()
            if not question:
                raise ValueError(f"Custom data row {idx} has empty question.")

            gt = item.get("ground_truth", item.get("answer", ""))
            if isinstance(gt, list):
                gt_norm = [str(x).strip() for x in gt if str(x).strip()]
            elif gt is None:
                gt_norm = []
            else:
                s_gt = str(gt).strip()
                gt_norm = [s_gt] if s_gt else []

            normalized = dict(item)
            normalized["Question"] = question
            normalized["question"] = question
            normalized["ground_truth"] = gt_norm
            normalized_rows.append(normalized)
        filtered_data = normalized_rows

    # ---------------------- Batch Generation Function ----------------------
    def generate_webpage_to_reasonchain_batch(
        original_questions: List[str],
        prev_reasonings: List[str],
        search_queries: List[str],
        documents: List[str],
        dataset_name: str,
        batch_output_records: List[Dict],  # New parameter to collect outputs
        max_tokens: int = 32768,
        coherent: bool = False,
    ) -> List[str]:
        user_prompts = [
            get_webpage_to_reasonchain_instruction(r, sq, doc)
            for r, sq, doc in zip(prev_reasonings, search_queries, documents)
        ]

        prompts = [{"role": "user", "content": up} for up in user_prompts]
        prompts = [tokenizer.apply_chat_template([p], tokenize=False, add_generation_prompt=True) for p in prompts]

        output = llm.generate(
            prompts,
            sampling_params=SamplingParams(
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.05,
            )
        )

        raw_outputs = [out.outputs[0].text for out in output]
        extracted_infos = [extract_answer(raw, mode='infogen') for raw in raw_outputs]

        for i, (p, r, e) in enumerate(zip(prompts, raw_outputs, extracted_infos)):
            batch_output_records.append({
                'prompt': p,
                'raw_output': r,
                'extracted_info': e
            })

        return extracted_infos

    # ---------------------- Preparation of Input Prompts ----------------------
    input_list = []
    for item in filtered_data:
        question = item['Question']

        if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
            if dataset_name in ['nq', 'triviaqa']:
                instruction = get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT)
            elif dataset_name in ['hotpotqa', 'musique', 'bamboogle', '2wiki']:
                instruction = get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT)
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_openqa(question)

        elif dataset_name in ['math500', 'aime', 'amc']:
            instruction = get_math_search_o1_instruction(MAX_SEARCH_LIMIT)
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_math(question)

        elif dataset_name == 'gpqa':
            instruction = get_gpqa_search_o1_instruction(MAX_SEARCH_LIMIT)
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
            elif 'llama' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(question)

        elif dataset_name == 'livecode':
            instruction = get_code_search_o1_instruction(MAX_SEARCH_LIMIT)
            question_title = item.get('question_title', '')
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
            else:
                user_prompt = get_task_instruction_code(question)
        else:
            user_prompt = ""  # Default to empty if dataset not matched

        raw_user_prompt = instruction + user_prompt
        item["_initial_user_prompt"] = raw_user_prompt
        prompt = [{"role": "user", "content": raw_user_prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_list.append(prompt)

    if subset_num != -1:
        input_list = input_list[:subset_num]
        filtered_data = filtered_data[:subset_num]

    # Initialize active sequences
    active_sequences = []
    for idx, (item, prompt) in enumerate(zip(filtered_data, input_list)):
        raw_question = item.get("Question", item.get("question", ""))
        init_user_prompt = item.get("_initial_user_prompt", raw_question)
        seq_messages = [{"role": "user", "content": str(init_user_prompt)}]
        seq_controller = controller_factory() if runtime_controller_enabled and controller_factory is not None else None
        if seq_controller is not None and hasattr(seq_controller, "reset_episode"):
            try:
                seq_controller.reset_episode()
            except Exception:
                pass
        active_sequences.append(
            {
                'item': item,
                'prompt': prompt,
                'output': '',
                'finished': False,
                'history': [],
                'messages': seq_messages,
                'search_count': 0,
                'executed_search_queries': set(),
                'step_records': [],
                'intervention_log': [],
                'step_id': 0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'controller_forced_answer': None,
                'controller': seq_controller,
            }
        )

    # ---------------------- Set Max Tokens ----------------------
    if 'qwq' in model_path.lower():
        if dataset_name in ['aime', 'amc', 'livecode']:
            max_tokens = 32768
        else:
            max_tokens = 20480
    else:
        max_tokens = 8192

    # ---------------------- Generation Function ----------------------
    def run_generation(sequences: List[Dict], max_tokens: int) -> List:
        prompts = [s['prompt'] for s in sequences]
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k_sampling,
            repetition_penalty=repetition_penalty,
            stop=[END_SEARCH_QUERY, tokenizer.eos_token],
            include_stop_str_in_output=True,
        )
        output_list = llm.generate(prompts, sampling_params=sampling_params)
        return output_list

    def rebuild_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def estimate_tokens_from_output(out_obj) -> Tuple[int, int]:
        prompt_token_ids = getattr(out_obj, "prompt_token_ids", None)
        tokens_input = len(prompt_token_ids) if isinstance(prompt_token_ids, list) else 0
        tokens_output = 0
        try:
            tokens_output = len(out_obj.outputs[0].token_ids)
        except Exception:
            tokens_output = 0
        return tokens_input, tokens_output

    # Function to extract text between two tags
    def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def replace_recent_steps(origin_str, replace_str):
        """
        Replaces specific steps in the original reasoning steps with new steps.
        If a replacement step contains "DELETE THIS STEP", that step is removed.

        Parameters:
        - origin_str (str): The original reasoning steps.
        - replace_str (str): The steps to replace or delete.

        Returns:
        - str: The updated reasoning steps after applying replacements.
        """

        def parse_steps(text):
            """
            Parses the reasoning steps from a given text.

            Parameters:
            - text (str): The text containing reasoning steps.

            Returns:
            - dict: A dictionary mapping step numbers to their content.
            """
            step_pattern = re.compile(r"Step\s+(\d+):\s*")
            steps = {}
            current_step_num = None
            current_content = []

            for line in text.splitlines():
                step_match = step_pattern.match(line)
                if step_match:
                    # If there's an ongoing step, save its content
                    if current_step_num is not None:
                        steps[current_step_num] = "\n".join(current_content).strip()
                    current_step_num = int(step_match.group(1))
                    content = line[step_match.end():].strip()
                    current_content = [content] if content else []
                else:
                    if current_step_num is not None:
                        current_content.append(line)
            
            # Save the last step if any
            if current_step_num is not None:
                steps[current_step_num] = "\n".join(current_content).strip()
            
            return steps

        # Parse the original and replacement steps
        origin_steps = parse_steps(origin_str)
        replace_steps = parse_steps(replace_str)

        # Apply replacements
        for step_num, content in replace_steps.items():
            if "DELETE THIS STEP" in content:
                # Remove the step if it exists
                if step_num in origin_steps:
                    del origin_steps[step_num]
            else:
                # Replace or add the step
                origin_steps[step_num] = content

        # Sort the steps by step number
        sorted_steps = sorted(origin_steps.items())

        # Reconstruct the reasoning steps as a single string
        new_reasoning_steps = "\n\n".join([f"{content}" for num, content in sorted_steps])

        return new_reasoning_steps

    # ---------------------- Initialize Collection Structure ----------------------
    # Initialize a list to collect batch outputs
    batch_output_records = []

    start_time = time.time()
    turn = 0

    # [Controller Integration] Apply controller decision after each complete step.
    def maybe_apply_controller(seq: Dict, step_data: Dict, default_finish: Optional[bool] = None) -> None:
        # Always record steps in output trajectory schema.
        seq['step_records'].append(step_data)
        seq['total_input_tokens'] += int(step_data.get("tokens_input", 0) or 0)
        seq['total_output_tokens'] += int(step_data.get("tokens_output", 0) or 0)
        if default_finish is not None:
            seq['finished'] = default_finish

        if not runtime_controller_enabled or seq.get('controller') is None:
            return
        try:
            ctrl_result = seq['controller'].process_step(step_data)
        except Exception as e:
            seq['intervention_log'].append(
                {"step": int(step_data.get("step_id", -1)), "action": "controller_error", "error": str(e)}
            )
            return

        new_messages, stop_answer, action_name = apply_controller_result(ctrl_result, seq['messages'])
        if action_name != "continue":
            seq['intervention_log'].append({"step": int(step_data.get("step_id", -1)), "action": action_name})

        if action_name == "stop":
            answer = stop_answer or ""
            seq['controller_forced_answer'] = answer
            seq['output'] += f"\n\\boxed{{{answer}}}\n"
            seq['messages'].append({"role": "assistant", "content": f"\\boxed{{{answer}}}"})
            seq['finished'] = True
            return

        if action_name in {"compress", "redirect", "mode_switch"}:
            seq['messages'] = new_messages
            seq['prompt'] = rebuild_prompt_from_messages(seq['messages'])
            seq['finished'] = False
            return

    # Main loop until all sequences are finished or maximum turns reached
    while True:
        # Identify sequences that need generation
        sequences_needing_generation = [seq for seq in active_sequences if not seq['finished']]

        if sequences_needing_generation:
            turn += 1
            print(f'\n-------------- Turn {turn} --------------')
            print(f"We have {len(sequences_needing_generation)} sequences needing generation...")
            outputs = run_generation(sequences_needing_generation, max_tokens)
            print("Generation completed, processing outputs...")

            # Initialize batch variables
            batch_relevant_info = []
            batch_original_questions = []
            batch_prev_reasonings = []
            batch_search_queries = []
            batch_documents = []
            batch_sequences = []

            # Collect URLs to fetch across all sequences
            all_urls_to_fetch = set()
            url_snippets = {}
            url_sequence_map = {}  # Map URL to list of sequences needing it

            # Process each sequence and collect URLs
            for seq, out in zip(sequences_needing_generation, outputs):
                text = out.outputs[0].text
                tokens_input, tokens_output = estimate_tokens_from_output(out)
                seq['history'].append(text)
                # Append generated text to prompt and output
                seq['prompt'] += text
                seq['output'] += text
                seq['messages'].append({"role": "assistant", "content": text})

                # Extract search query
                search_query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

                # If a search query is present and needs to be executed
                if search_query and seq['output'].rstrip().endswith(END_SEARCH_QUERY):
                    if seq['search_count'] < MAX_SEARCH_LIMIT and search_query not in seq['executed_search_queries']:
                        # Execute search, use cache if available
                        if search_query in search_cache:
                            results = search_cache[search_query]
                            print(f"Using cached search results for query: \"{search_query}\"")
                        else:
                            try:
                                if search_backend == 'duckduckgo':
                                    results = web_search(search_query, backend='duckduckgo', max_results=top_k)
                                else:
                                    results = web_search(search_query, backend='bing', subscription_key=bing_subscription_key, endpoint=bing_endpoint, market='en-US', language='en')
                                search_cache[search_query] = results
                                print(f"Executed and cached search for query: \"{search_query}\"")
                            except Exception as e:
                                print(f"Error during search query '{search_query}': {e}")
                                search_cache[search_query] = {}
                                results = {}

                        # Extract relevant information from Bing search results
                        relevant_info = extract_relevant_info(results)[:top_k]
                        seq['relevant_info'] = relevant_info

                        # Extract URLs and snippets
                        urls_to_fetch = [it['url'] for it in relevant_info]
                        snippets = {info['url']: info['snippet'] for info in relevant_info if 'snippet' in info}

                        # Filter URLs that are not cached
                        urls_to_fetch_filtered = [u for u in urls_to_fetch if u not in url_cache]
                        cached_urls = [u for u in urls_to_fetch if u in url_cache]

                        # Store info for all_urls_to_fetch and url_snippets
                        for url in urls_to_fetch_filtered:
                            all_urls_to_fetch.add(url)
                            url_snippets[url] = snippets.get(url, "")

                        all_reasoning_steps = seq['output']
                        all_reasoning_steps = all_reasoning_steps.replace('\n\n', '\n').split("\n")

                        truncated_prev_reasoning = ""
                        for i, step in enumerate(all_reasoning_steps):
                            truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"

                        prev_steps = truncated_prev_reasoning.split('\n\n')
                        if len(prev_steps) <= 5:
                            truncated_prev_reasoning = '\n\n'.join(prev_steps)
                        else:
                            truncated_prev_reasoning = ''
                            for i, step in enumerate(prev_steps):
                                if i == 0 or i >= len(prev_steps) - 4 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                                    truncated_prev_reasoning += step + '\n\n'
                                else:
                                    if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                                        truncated_prev_reasoning += '...\n\n'
                        truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')

                        # Collect parameters for batch processing
                        batch_relevant_info.append(relevant_info)
                        batch_original_questions.append(seq['item']['Question'])
                        batch_prev_reasonings.append(truncated_prev_reasoning)
                        batch_search_queries.append(search_query)
                        batch_sequences.append(seq)
                        # [Controller Integration] Search step waits until observation is appended.
                        seq['_pending_controller_step'] = {
                            "step_id": int(seq['step_id']),
                            "thought": text,
                            "action_type": "search",
                            "action_input": {"query": search_query},
                            "observation": "",
                            "tokens_input": int(tokens_input),
                            "tokens_output": int(tokens_output),
                        }
                        seq['step_id'] += 1

                        # Update search count and executed queries
                        seq['search_count'] += 1
                        seq['executed_search_queries'].add(search_query)

                    elif seq['search_count'] >= MAX_SEARCH_LIMIT:
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        seq['messages'].append({"role": "tool", "content": limit_message})
                        # [Controller Integration] Search-limit observation available immediately.
                        step_data = {
                            "step_id": int(seq['step_id']),
                            "thought": text,
                            "action_type": "search",
                            "action_input": {"query": search_query},
                            "observation": limit_message,
                            "tokens_input": int(tokens_input),
                            "tokens_output": int(tokens_output),
                        }
                        seq['step_id'] += 1
                        maybe_apply_controller(seq, step_data, default_finish=False)
                        print(f"Search limit reached for query: \"{search_query}\"")

                    elif search_query in seq['executed_search_queries']:
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        seq['messages'].append({"role": "tool", "content": limit_message})
                        # [Controller Integration] Repeated-query observation available immediately.
                        step_data = {
                            "step_id": int(seq['step_id']),
                            "thought": text,
                            "action_type": "search",
                            "action_input": {"query": search_query},
                            "observation": limit_message,
                            "tokens_input": int(tokens_input),
                            "tokens_output": int(tokens_output),
                        }
                        seq['step_id'] += 1
                        maybe_apply_controller(seq, step_data, default_finish=False)
                        print(f"Repeated search for query: \"{search_query}\"")

                else:
                    # No search query -> treat as respond step.
                    # [Controller Integration] Respond step with empty observation.
                    step_data = {
                        "step_id": int(seq['step_id']),
                        "thought": text,
                        "action_type": "respond",
                        "action_input": {"content": text},
                        "observation": "",
                        "tokens_input": int(tokens_input),
                        "tokens_output": int(tokens_output),
                    }
                    seq['step_id'] += 1
                    maybe_apply_controller(seq, step_data, default_finish=True)
                    print("Sequence marked as complete.")

            # Batch fetch all URLs at once to optimize speed
            if all_urls_to_fetch:
                print(f"Fetching {len(all_urls_to_fetch)} URLs...")
                try:
                    fetched_contents = fetch_page_content(
                        list(all_urls_to_fetch),
                        use_jina=use_jina,
                        jina_api_key=jina_api_key,
                        # snippets=url_snippets  # Do not pass snippets when updating url_cache directly
                    )
                    print(f"Fetched {len(fetched_contents)} URLs successfully.")
                except Exception as e:
                    print(f"Error during batch URL fetching: {e}")
                    fetched_contents = {url: f"Error fetching URL: {e}" for url in all_urls_to_fetch}
                # Update cache with fetched contents
                for url, content in fetched_contents.items():
                    url_cache[url] = content

            # After fetching, prepare formatted documents for batch processing
            for relevant_info in batch_relevant_info:
                formatted_documents = ""
                for i, doc_info in enumerate(relevant_info):
                    url = doc_info['url']
                    raw_context = url_cache.get(url, "")
                    doc_info['snippet'] = doc_info['snippet'].replace('<b>','').replace('</b>','')            
                    success, filtered_context = extract_snippet_with_context(raw_context, doc_info['snippet'], context_chars=max_doc_len)
                    if success:
                        context = filtered_context
                    else:
                        context = raw_context[:max_doc_len*2]

                    doc_info['context'] = context
                    formatted_documents += f"**Web Page {i + 1}:**\n"
                    formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
                    
                batch_documents.append(formatted_documents)

            # After fetching, prepare for batch processing if there are any
            if batch_sequences:
                print(f"Batch processing {len(batch_sequences)} sequences with generate_webpage_to_reasonchain_batch...")
                webpage_analyses = generate_webpage_to_reasonchain_batch(
                    original_questions=batch_original_questions,
                    prev_reasonings=batch_prev_reasonings,
                    search_queries=batch_search_queries,
                    documents=batch_documents,
                    dataset_name=dataset_name,
                    batch_output_records=batch_output_records,  # Pass the collection list
                    max_tokens=max_tokens,
                )
                print("Batch generation completed, assigning outputs to sequences...")

                for seq, analysis in zip(batch_sequences, webpage_analyses):
                    if isinstance(analysis, str):
                        append_text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)
                        seq['messages'].append({"role": "tool", "content": append_text})
                    else:
                        append_text = replace_recent_steps(seq['output'], analysis)
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)
                        seq['messages'].append({"role": "tool", "content": append_text})

                    # [Controller Integration] Insert controller exactly after observation append.
                    pending = seq.pop('_pending_controller_step', None)
                    if pending is not None:
                        pending["observation"] = append_text
                        maybe_apply_controller(seq, pending, default_finish=False)

            # [Controller Integration] Safety net: flush any unconsumed pending steps.
            # This handles cases where batch fetch failed or sequence wasn't included in batch.
            for seq in active_sequences:
                pending = seq.pop('_pending_controller_step', None)
                if pending is not None:
                    pending["observation"] = pending.get("observation", "") or "[fetch failed or skipped]"
                    maybe_apply_controller(seq, pending, default_finish=False)

        # Check if all sequences are finished
        unfinished = [seq for seq in active_sequences if not seq['finished']]
        if not unfinished:
            break
        else:
            if turn >= MAX_TURN:
                print(f"Maximum number of turns ({MAX_TURN}) reached, stopping.")
                break

    total_time = time.time() - start_time

    # [Controller Integration] Attach per-trajectory details for output compatibility.
    for idx, seq in enumerate(active_sequences):
        item = seq['item']
        item.pop("_initial_user_prompt", None)
        item['task_id'] = item.get('task_id', idx)
        item['question'] = item.get('question', item.get('Question', ''))
        item['ground_truth'] = item.get('ground_truth', item.get('answer', item.get('Correct Choice', '')))
        item['agent_answer'] = seq.get('controller_forced_answer', None)
        item['total_steps'] = int(len(seq['step_records']))
        item['total_input_tokens'] = int(seq['total_input_tokens'])
        item['total_output_tokens'] = int(seq['total_output_tokens'])
        item['total_time'] = float(total_time / max(len(active_sequences), 1))
        item['steps'] = seq['step_records']
        item['intervention_log'] = seq['intervention_log']
        base_cfg = item.get('config', {})
        if not isinstance(base_cfg, dict):
            base_cfg = {"raw_config": str(base_cfg)}
        base_cfg.update(
            {
                "controller_enabled": bool(runtime_controller_enabled),
                "baseline_type": baseline_type,
                "gate_threshold": float(gate_threshold),
                "controller_path": controller_path if runtime_controller_enabled else None,
                "ref_dist_path": ref_dist_path if runtime_controller_enabled else None,
            }
        )
        item['config'] = base_cfg

    # ---------------------- Save Batch Output Records to JSON File ----------------------
    # Define output JSON file path
    t = time.localtime()
    batch_output_file = os.path.join(output_dir, f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.info_extract.json')

    # Save batch_output_records to JSON file
    with open(batch_output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_output_records, f, ensure_ascii=False, indent=2)

    print(f"Batch outputs saved to {batch_output_file}")

    # Prepare output list for evaluation
    output_list = [seq['output'] for seq in active_sequences]

    # Run evaluation (keeps original evaluator; added fields remain in result JSON)
    run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split)

    # ---------------------- Update Search and URL Cache ----------------------
    print('Updating Search and URL Cache...')
    # Load existing caches or initialize empty dictionaries
    if os.path.exists(search_cache_path):
        with open(search_cache_path, 'r', encoding='utf-8') as f:
            search_cache_new = json.load(f)
    else:
        search_cache_new = {}

    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache_new = json.load(f)
    else:
        url_cache_new = {}

    search_cache.update(search_cache_new)
    url_cache.update(url_cache_new)

    save_caches()

    print("Process completed.")

if __name__ == "__main__":
    main()
