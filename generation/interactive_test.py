from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model_name', help="Model Name ('gpt2','gpt2-medium','gpt2-large','gpt2-xl') or a finetuned model name", type=str, required=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name).cuda()

while(True):
    inputs = input("Input a prompt for the model: ")
    input_tensor = tokenizer.encode(inputs, return_tensors='pt').to(model.device)
    outputs = model.generate(input_tensor,
                             do_sample=True,
                             top_p=0.7,
                             repetition_penalty=1.2,
                             pad_token_id=tokenizer.eos_token_id,
                             max_length=256)
    out = [tokenizer.decode(x) for x in outputs][0]
    print("Output:\n" + out)
    print("---------------------------\n")
