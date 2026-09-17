import os
import numpy as np
import time
import logging
from datetime import datetime
from agent_system.environments.env_manager import *
from openai import OpenAI
import yaml
from omegaconf import OmegaConf
import json 
from transformers import AutoTokenizer, AutoModelForCausalLM
def build_env(env_name, env_num=1):
    group_n = 1
    if env_name == "alfworld":
        # Test AlfWorldEnvironmentManager
        from agent_system.environments.env_package.alfworld import alfworld_projection
        from agent_system.environments.env_package.alfworld import build_alfworld_envs
        alf_config_path = os.path.join(os.path.dirname(__file__), '../../agent_system/environments/env_package/alfworld/configs/config_tw.yaml')
        resources_per_worker = {"num_cpus": 0.1} 
        config = OmegaConf.load(alf_config_path)

        envs = build_alfworld_envs(alf_config_path, seed=1, env_num=env_num, group_n=group_n, is_train=False, resources_per_worker=resources_per_worker)

        env_manager = AlfWorldEnvironmentManager(envs, alfworld_projection, config)
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")
    
    return env_manager

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def generate_action(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=True, temperature=0.7)
    action = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return action


if __name__ == "__main__":

    # -------- logging ----------
    os.makedirs("logs/alfworld", exist_ok=True)
    log_fp = os.path.join(
        "logs/alfworld", f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )

    # -------- Parameters ----------
    max_steps = 50
    env_num = 1
    test_times = 1
    env_name = "alfworld" 
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    # Keywords for 6 subtasks
    TASKS = [
        "pick_and_place",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_clean_then_place_in_recep",
    ]

    # -------- Environment and agent setup ----------
    env_manager = build_env(env_name, env_num)

    tokenizer, model = load_model(model_name)

    # Accumulated statistics
    overall_success_rates = []         # Overall success per round
    task_success_history = defaultdict(list)  # Subtask success per round

    # ======================= Main Loop =======================
    for test_idx in range(test_times):
        logging.info(f"\n========== Start test {test_idx} ==========")
        start_time = time.time()

        obs, infos = env_manager.reset()
        env_dones = [False] * env_num

        # Statistics for single round
        overall_success_this_round = np.zeros(env_num, dtype=bool)
        task_success_cnt = defaultdict(int)
        task_total_cnt = defaultdict(int)


        for step_idx in range(max_steps):
            logging.info(f"Step {step_idx}; Dones ({np.array(env_dones).sum().item()}/{env_num}); SR {overall_success_this_round.mean().item()}")

            # --- Assemble actions ---
            actions = []
            for i in range(env_num):
                if env_dones[i]:
                    actions.append("None")
                else:
                    # print(obs["text"][i])
                    # action = input("Enter action: ")
                    # action = "<action>" + action + "</action>"
                    action = generate_action(model, tokenizer, obs["text"][i])
                    actions.append(action.split("</think>")[1])

            # --- Environment stepping ---

            obs, rewards, dones, infos = env_manager.step(actions)

            # --- Determine endings and successes ---
            for i in range(env_num):
                if env_dones[i]:
                    continue

                if dones[i]:
                    env_dones[i] = True
                    won = bool(infos[i].get("won", False))
                    overall_success_this_round[i] = won

                    # Parse task type
                    gamefile = infos[i].get("extra.gamefile", "")
                    matched = False
                    for task in TASKS:
                        if task in gamefile:
                            task_total_cnt[task] += 1
                            if won:
                                task_success_cnt[task] += 1
                            matched = True
                            break
                    if not matched:
                        # Unrecognized tasks are also counted in total
                        task_total_cnt["other"] += 1
                        if won:
                            task_success_cnt["other"] += 1

            if all(env_dones):
                logging.info("All environments finished early!")
                break

        # -------- Single round results --------
        round_success_rate = overall_success_this_round.mean()
        overall_success_rates.append(round_success_rate)

        logging.info(f"Test {test_idx} overall success: {round_success_rate:.4f}")

        for task in TASKS + ["other"]:
            if task_total_cnt.get(task, 0) > 0:
                rate = task_success_cnt[task] / task_total_cnt[task]
                task_success_history[task].append(rate)
                logging.info(
                    f"    {task:<35s}: {rate:.4f} "
                    f"({task_success_cnt[task]}/{task_total_cnt[task]})"
                )

        logging.info(
            f"Test {test_idx} time elapsed: {time.time() - start_time:.2f}s\n"
        )

    # ======================= Final Summary =======================
    logging.info("=============== Final Summary ===============")
    logging.info(
        f"Total tests: {test_times} | Envs / test: {env_num} | Total envs: {env_num * test_times}"
    )
    logging.info(
        f"Overall success avg ± std: "
        f"{np.mean(overall_success_rates):.4f} ± {np.std(overall_success_rates):.4f}"
    )

    for task in TASKS + ["other"]:
        if task_success_history.get(task):
            logging.info(
                f"{task:<35s}: "
                f"{np.mean(task_success_history[task]):.4f} ± "
                f"{np.std(task_success_history[task]):.4f}"
            )
