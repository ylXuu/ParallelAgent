from inference.self_manager.agent import run_one
from tqdm import tqdm
import json
import asyncio
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    agent_name = args.agent_name
    model_name = args.model_name
    dataset = args.dataset
    
    if agent_name == "self_manager":
        config = {
            "api_key": "token-abc123",
            "base_url": "http://127.0.0.1:30000/v1",
            "timeout": 30,
            "model": model_name,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 32768,
            "context_window": 131072,
            "max_rounds": 50,
            "child_max_rounds": 50,
            "max_concurrent_branches": 30,
            "summarize_api_key": "your_api_key",
            "summarize_base_url": "your_base_url",
            "summarize_model": "your_model",
        }

    else:
        raise ValueError(f"Invalid agent name: {agent_name}")


    if dataset == "deepresearch_bench":
        data_path = "path/to/query.jsonl"
        output_path = f"path/to/output.jsonl"
        
        data_list = []
        with open(data_path, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                data_list.append({
                    "item": {
                        "question": data["prompt"],
                        "answer": ""
                    },
                    "id": data["id"],
                })
        
        output_list = []
        for data in tqdm(data_list):
            answer, stats, msgs = asyncio.run(run_one(data["item"]["question"], config))
            result = {
                "prediction": answer,
                "messages": {
                    "main": msgs["main"],
                    "child": msgs["child"],
                    "stats": stats,
                }
            }
            
            output_list.append({
                "id": data["id"],
                "prompt": data["item"]["question"],
                "article": result["prediction"],
                "messages": result["messages"],
            })
        
        with open(output_path, "w", encoding="utf-8") as file:
            for item in output_list:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Invalid dataset: {dataset}")


