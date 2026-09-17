import os
import numpy as np
import time
import logging
from datetime import datetime
from agent_system.environments.env_manager import *
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import json


def build_env(env_name, env_num=1):
    group_n = 1
    if env_name == "alfworld":
        # Test AlfWorldEnvironmentManager
        from agent_system.environments.env_package.alfworld import alfworld_projection
        from agent_system.environments.env_package.alfworld import build_alfworld_envs
        alf_config_path = os.path.join(os.path.dirname(__file__), '../../agent_system/environments/env_package/alfworld/configs/config_tw.yaml')
        resources_per_worker = {"num_cpus": 0.1} 
        envs = build_alfworld_envs(alf_config_path, seed=1, env_num=env_num, group_n=group_n, is_train=False, resources_per_worker=resources_per_worker)
        env_manager = AlfWorldEnvironmentManager(envs, alfworld_projection, 'alfworld/AlfredThorEnv')
    elif env_name == "webshop":
        from agent_system.environments.env_package.webshop import webshop_projection
        from agent_system.environments.env_package.webshop import build_webshop_envs
        file_path = os.path.join(os.path.dirname(__file__), '../../agent_system/environments/env_package/webshop/webshop/data/items_shuffle_1000.json')
        attr_path = os.path.join(os.path.dirname(__file__), '../../agent_system/environments/env_package/webshop/webshop/data/items_ins_v2_1000.json')
        webshop_config_path = os.path.join(os.path.dirname(__file__), '../../agent_system/environments/env_package/webshop/webshop/data/ppo_trainer.yaml')
        resources_per_worker = {"num_cpus": 0.1} 
        config = OmegaConf.load(webshop_config_path)
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        envs = build_webshop_envs(seed=config.env.seed, env_num=env_num, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        env_manager = WebshopEnvironmentManager(envs, webshop_projection, config)
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")
    
    return env_manager


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def generate_action(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=2048, num_beams=5, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":

    # -------- logging ----------
    os.makedirs("logs/webshop", exist_ok=True)
    log_fp = os.path.join(
        "logs/webshop", f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )

    # -------- Parameters ----------
    max_steps = 15
    env_num = 1
    test_times = 1
    env_name = "webshop" 

    # -------- Environment and agent setup ----------
    env_manager = build_env(env_name, env_num)

    # Accumulated statistics
    overall_success_rates = []         # Overall success per round
    task_success_history = defaultdict(list)  # Subtask success per round

    # -------- Model setup ----------
    tokenizer, model = load_model("Qwen/Qwen2.5-1.5B-Instruct")

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

        for task in ["other"]:
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

    for task in ["other"]:
        if task_success_history.get(task):
            logging.info(
                f"{task:<35s}: "
                f"{np.mean(task_success_history[task]):.4f} ± "
                f"{np.std(task_success_history[task]):.4f}"
            )
