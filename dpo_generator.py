from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8008, type=int, help="Port number to run the Flask app")
    parser.add_argument("--gpu", default=0, type=int, help="GPU device index to use")
    return parser.parse_args()

args = parse_arguments()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Initialize Flask app
app = Flask(__name__)

# Model and Tokenizer Initialization
MODEL_NAME = "cerebras/btlm-3b-8k-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device).eval()  # Set model to evaluation mode
model = model.half()

@app.route('/', methods=['POST'])
def generate():
    content = request.json
    prompt = content.get('prompt', '')

    # Hard-coded parameters for controlling the generation
    temperature = 0.7
    top_k = 50
    top_p = 0.95

    if not prompt:
        return jsonify(error="Prompt not provided"), 400

    # Tokenize the prompt with truncation
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)

    # Use torch.no_grad() to save memory during inference
    with torch.no_grad():
        completions = model.generate(
            input_ids, 
            max_length=min(len(input_ids[0]) + 200, tokenizer.model_max_length),  # Adjust for desired completion length
            num_return_sequences=2, 
            pad_token_id=tokenizer.eos_token_id, 
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

    # Decode completions excluding the input prompt
    completions_texts = [tokenizer.decode(completion[len(input_ids[0]):], skip_special_tokens=False) for completion in completions]

    # Structure the response
    response_data = {
        "prompt": prompt,
        "responses": completions_texts
    }

    # Explicitly delete tensors and force garbage collection
    del input_ids
    del completions
    gc.collect()
    torch.cuda.empty_cache()

    return jsonify(response_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, threaded=True, debug=False)
