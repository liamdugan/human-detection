import json, argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('-f','--file', help="The path to the file/folder you want to analyze", type=str)

args = parser.parse_args()

with open(args.file, 'r') as f:
    data = json.load(f)

    # Check the number of generations done
    num_gens = len(data['generations'])

    # Initialize a counter
    prompt_len_cnt = Counter([len(gen['prompt']) for gen in data['generations']])
    p_value_cnt = Counter([gen['p'] for gen in data['generations']])
    both_cnt = Counter([(len(gen['prompt']),gen['p']) for gen in data['generations']])

    total_num_issues = 0
    total_num_all_human = 0
    
    for gen in data['generations']:
      prompt_len = len(gen['prompt'])
      gen_len = len(gen['generation'])
      if prompt_len + gen_len < 10:
        total_num_issues += 1
        if gen_len == 0:
          total_num_all_human += 1

    print("Analysis results for file: " + str(args.file))
    print("-------------------------")
    print("Total Number of Generations: " + str(num_gens))
    print("-------------------------")
    print("Total Number of Generations that don't add up to 10: " + str(total_num_issues))
    print("Total Number of All Human Generations that don't add to 10: " + str(total_num_all_human))
    print("% bugged generations: " + str(float(total_num_issues) / float(num_gens)))
    print("-------------------------")
    print("Expected number of Generations per prompt length: " + str(num_gens/10))
    print("Expected number of Generations per p value: " + str(num_gens/11))
    print("Expected number of Generations per prompt & p: " + str(num_gens/110))
    print("-------------------------")
    print("Breakdown by prompt length (length, #gens)")
    print(sorted(prompt_len_cnt.items()))
    print("Breakdown by p value (p value, #gens)")
    print(sorted(p_value_cnt.items()))

    print("-------------------------")
    worst_len = min(prompt_len_cnt.items(), key=lambda x:x[1])
    print("Worst Prompt Length is length " + str(worst_len[0]) + " with " + str(worst_len[1]) + " generations")

    worst_p = min(p_value_cnt.items(), key=lambda x:x[1])
    print("Worst P Value is " + str(worst_p[0]) + " with " + str(worst_p[1]) + " generations")

    worst_both = min(both_cnt.items(), key=lambda x:x[1])
    print("Worst Length + P value combo is " + str(worst_both[0]) + " with " + str(worst_both[1]) + " generations")

    print("-------------------------")
    vary_p_yield = (float(worst_both[1] * 110.0) / float(num_gens)) * 100
    print("If Varying P, we can use {0} generations per p_value length combo ({1}% Yield)".format(str(worst_both[1]), str(int(vary_p_yield))))

    len_yield = (float(worst_len[1] * 10.0) / float(num_gens)) * 100
    print("If Not Varying P, we can use {0} generations per length ({1}% Yield)".format(str(worst_len[1]), str(int(len_yield))))
